import numpy as np
import torch
import torch.nn as nn
import random
import argparse
import os
import evaluate
from tqdm import tqdm
from torch.utils.data import Sampler
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.distributed import is_initialized, get_rank

import deepspeed
import sys
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer

# from utils.optim import LinearWarmupCosineLRScheduler, cosine_schedule_with_min_lr
from torch.optim.lr_scheduler import LambdaLR

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modeling.Qwen3_FastSLM_na_downsampler import FastALM

from data_loader.ASR_loader import (
    Fleurs,
    Voxpopuli, 
    LibriSpeech
)
from data_loader.load_dataset_Qwen3_v2 import (
    LOAD_DATASET, 
    SLLMDataCollatorWhithPadding
)

import warnings

warnings.filterwarnings('ignore')

from prettytable import PrettyTable

def check_nan(tensor, name=""):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN/Inf detected in {name}!")
        return True
    return False

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    # print(table)
    print(f"Total Trainable Params: {total_params* 1e-6:.3f}M")
    return total_params

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

class Trainer:
    def __init__(self,args):
        super(Trainer,self).__init__()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._build_model()


    def seed_everything(self,seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if self.args.multi_gpu:
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _build_model(self):
        model = FastALM(
            embed_dim=self.args.embed_dim,
            speech_dim=self.args.speech_dim
        )

        count_parameters(model)
        self.checkpoint = None if self.args.check_point == None else torch.load(self.args.check_point,map_location='cpu')
        if self.checkpoint is not None:
            model.load_state_dict(self.checkpoint['model_state_dict'])
            # model.llm.load_state_dict(self.checkpoint['llm'])
            # model.encoder.load_state_dict(self.checkpoint['encoder'])
        
        model.llm.gradient_checkpointing_enable()

        model = model.to(self.device)
        return model
        
    
    def load_data(self):
        train_dataset = LOAD_DATASET(
            roots=self.args.data_path,
            # iteration=int(300000 * self.args.batch_size) if self.checkpoint else 0
            iteration=int((300000 + self.checkpoint['iteration']) * self.checkpoint['batch_size']) if self.checkpoint else 0,
        )
       
        train_loader=torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True,
            collate_fn=SLLMDataCollatorWhithPadding()
        )
        return train_loader
    
    def train(
        self,
        train_loader,
        epoch,
        ):
        total_loss=0
        iter_loss=0
        eval_loss=0
        self.model.train()
        idx=1

        min_loss = torch.inf
        self.model.zero_grad()
        data_iter = iter(train_loader)
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{self.args.epoch}", unit="batches")

        batch_idx = 0
        while batch_idx < len(train_loader):
            try:
                data = next(data_iter)
            except Exception:
                print(f"⚠️ Skipping batch {batch_idx} due to data loading error")
                batch_idx += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': 0,
                    'lr':0
                })
                progress_bar.update(1)
                if (batch_idx+1) % self.args.acc_steps == 0:
                    self.scheduler.step()
                continue

            try:
                # Forward & backward
                audio = data['audio']
                input_ids = data['input_ids'].to(self.device)
                labels = data["labels"].to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast(dtype=self.args.dtype):
                        loss, _, labels = self.model(
                            audio,
                            input_ids,
                            labels,
                        )

                    total_loss += loss.item()
                    iter_loss += loss.item()
                    if self.args.multi_gpu:
                        self.model.backward(loss)
                    else:
                        self.scaler.scale(loss).backward()

                    if (batch_idx+1) % self.args.acc_steps == 0:
                        if self.args.multi_gpu:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                            self.model.step()
                            self.model.zero_grad()
                            self.scheduler.step()
                        else:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            self.scheduler.step()
                else:
                    loss, _, labels = self.model(
                        audio,
                        input_ids,
                        labels,
                    )
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'lr':f"{self.scheduler.get_last_lr()[0]:.7f}"
                })
        
                # model checkpoint each train step
                idx += 1

                if idx % self.args.check_point_step == 0 or idx == len(progress_bar) - 1:
                    self.save_checkpoint(iter_=batch_idx)

                if args.eval_step is not None:
                    if idx % self.args.eval_step == 0:
                        self.model.llm.gradient_checkpointing_disable()
                        clean_res, clean_ref, timings = self.get_decoding_result(self.test_loader,'en')
                        if is_main_process():
                            libri_loss = self.save_eval_log(idx,epoch,iter_loss,clean_ref,clean_res,'LibriSpeech test-clean')
                            print(f"LibriSpeech test-clean WER: {libri_loss * 100:.2f} %")
                        else:
                            temp_loss = self.save_eval_log(idx,epoch,iter_loss,clean_ref,clean_res,'LibriSpeech test-clean',result_only=True)
                            print(f"Worker 1 LibriSpeech test-clean WER: {temp_loss * 100:.2f} %")

                        clean_res, clean_ref, timings = self.get_decoding_result(self.voxpopuli_loader,'en')
                        if is_main_process():
                            vox_loss = self.save_eval_log(idx,epoch,iter_loss,clean_ref,clean_res,'Voxpopuli')
                            print(f"Voxpopuli WER: {vox_loss * 100:.2f} %")
                        else:
                            temp_loss = self.save_eval_log(idx,epoch,iter_loss,clean_ref,clean_res,'Voxpopuli',result_only=True)
                            print(f"Worker 1 Voxpopuli WER: {temp_loss * 100:.2f} %")

                        clean_res, clean_ref, timings = self.get_decoding_result(self.korean_loader,'ko')
                        if is_main_process():
                            kor_loss = self.save_eval_log(idx,epoch,iter_loss,clean_ref,clean_res,'Fleurs-Korean')
                            print(f"Fleurs-Korean WER: {kor_loss * 100:.2f} %")
                            iter_loss = 0
                        else:
                            temp_loss = self.save_eval_log(idx,epoch,iter_loss,clean_ref,clean_res,'Fleurs-Korean',result_only=True)
                            print(f"Worker 1 Fleurs-Korean WER: {temp_loss * 100:.2f} %")
                        
                        eval_loss = (libri_loss + vox_loss + kor_loss) / 3
                        if eval_loss < min_loss:
                            self.save_checkpoint(iter_=batch_idx,best=True)
                            print('Eval loss decreased {:.3f} ---> {:.3f}'.format(min_loss, eval_loss))
                            min_loss = eval_loss

                        self.model.llm.gradient_checkpointing_enable()

            except Exception as e:
                print(f"⚠️ Skipping batch {batch_idx} due to training error: {e}")

            batch_idx += 1

            progress_bar.update(1)
        progress_bar.close()
        total_loss /= len(progress_bar)
        return total_loss


    def save_checkpoint(self, iter_=None, best=False):
        if not is_main_process():
            return

        engine_model = self.model.module if hasattr(self.model, "module") else self.model
        
        with deepspeed.zero.GatheredParameters(engine_model.parameters(), modifier_rank=0):
            model_state_dict = engine_model.state_dict()
            torch.save({
                'iteration': iter_,
                'lr': self.scheduler.get_last_lr()[0],
                'batch_size':self.args.batch_size,
                'model_state_dict': model_state_dict
            }, f'model_weight/{self.args.model_name}_{iter_}_v{self.args.version}.pt' if not best else
            f'model_weight/Best_{self.args.model_name}_v{self.args.version}.pt')
            del engine_model, model_state_dict
    
    def save_eval_log(self,idx,epoch,iter_loss,clean_ref,clean_res,dataset_name,result_only=False):
        dummy_clean_ref = [ref for ref in clean_ref if ref.strip()]
        clean_res = [res for ref, res in zip(clean_ref, clean_res) if ref.strip()]
        clean_ref = dummy_clean_ref
        eval_loss = self.metrics_wer.compute(references=clean_ref, predictions=clean_res)
        if not result_only:
            with open(f"Results/{self.args.model_name}_v{self.args.version}.txt", 'a') as f:
                if idx == self.args.eval_step:
                    f.write('Configuration:{}\n'.format(self.args))
                f.write('Epoch: {} \tIteration:{} \tTrain Loss: {:.3f} \tWER: {:.3f}% \n'.format(
                    epoch, idx, iter_loss / self.args.eval_step, eval_loss * 100))
                f.write(f'Example of Generated {dataset_name} Text\n')
                for i in range(5):
                    f.write('Label: {}\n'.format(clean_ref[i]))
                    f.write('Generate: {}\n'.format(clean_res[i]))
                f.close()
        return eval_loss
        
    def get_decoding_result(self,loader,lang):
        if lang == 'en':
            normalizer = EnglishTextNormalizer()
        else:
            normalizer = BasicTextNormalizer()
        refs = []
        res = []
        timings = np.zeros((len(loader),1))
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.model.eval()
        for rep, b in enumerate(tqdm(loader)):
            audio = b['audio']
            input_ids = b['input_ids'].to(self.device)
            labels = b["labels"].to(self.device)
            # labels = b["labels"].long().to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.args.dtype):
                    starter.record()
                    results = self.model.generate(
                        input_ids,
                        audio,
                        max_new_tokens=self.args.max_new_tokens,
                        top_p=0.01,
                        top_k=0
                    )
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[rep] = curr_time
                    
            for r in results:
                res.append(normalizer(r).strip())
            
            for l in labels:
                l[l == -100] = self.model.tokenizer.eos_token_id
                ref = self.model.tokenizer.decode(l,skip_special_tokens=True)
                refs.append(normalizer(ref))
        self.model.train()
        return res, refs, timings

    def main(self):
        self.seed_everything(seed=0) 
        if not self.args.only_test:
            train_loader = self.load_data()
        
        voxpopuli_test = Voxpopuli('facebook/voxpopuli','test','en')
        
        # korean_test = Korean_ASR('korean_test.json','train','ko')
        korean_test = Fleurs('google/fleurs/ko_kr','test','ko')
        test_clean = LibriSpeech('sanchit-gandhi/librispeech-data','test.clean')

        self.voxpopuli_loader = torch.utils.data.DataLoader(
            dataset=voxpopuli_test,
            # batch_size=self.args.batch_size,
            batch_size=32,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=SLLMDataCollatorWhithPadding()
        )
        self.korean_loader = torch.utils.data.DataLoader(
            dataset=korean_test,
            # batch_size=self.args.batch_size,
            batch_size=32,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=SLLMDataCollatorWhithPadding()
        )
        self.test_loader = torch.utils.data.DataLoader(
                dataset=test_clean,
                batch_size=32,
                shuffle=False,
                num_workers=self.args.workers,
                collate_fn=SLLMDataCollatorWhithPadding()
            )
        train_loss = torch.zeros(self.args.epoch)
        
        self.scaler = GradScaler()    
        
        start_epoch = 0
       
        # checkpoint = torch.load(self.args.check_point,map_location='cuda: 0') if self.args.check_point else None
    
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.args.init_lr,weight_decay=self.args.decay)

        if self.args.use_scheduler and not self.args.only_test:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.args.max_lr if not self.args.check_point else self.checkpoint['lr'],
                # max_lr=self.args.max_lr,
                steps_per_epoch=len(train_loader)//self.args.acc_steps,
                epochs=self.args.epoch,
                pct_start=self.args.pct_start,
                anneal_strategy="linear"
            )
        else:
            self.scheduler = None

        self.metrics_wer = evaluate.load("wer")
        if self.args.know_distill:
            self.kl_loss = nn.KLDivLoss(reduction="none").to(self.device)

        if self.args.multi_gpu:
            ds_config = {
                "train_batch_size": self.args.batch_size,
                "gradient_accumulation_steps": 1,
                "bf16": {
                    "enabled": True if self.args.use_amp else False
                },
                "zero_optimization": {
                    "stage": self.args.zero_optimization
                }
            }

            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                config=ds_config
            )

        for e in range(start_epoch,self.args.epoch):
            train_loss[e] = self.train(
                train_loader=train_loader,
                epoch=e
            )
            print('Epoch: {} \tTraining Loss: {:.4f}'.format(e+1, train_loss[e]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='speech modality adaptation')
    args = parser.parse_args()
    # model config
    args.embed_dim=2560
    args.speech_dim=1280
    args.llm_path='LLM_repo_id'
    args.encoder_path='encoder_repo_id'

    # data config
    args.data_path=[
        'your_dataset'
    ]
    # Training config
    args.epoch=1
    args.batch_size=16
    args.workers=16
    args.check_point_step=40000
    args.eval_step=None
    args.max_grad_norm=1.0
    args.know_distill=False
    # optimizer and scheduler config
    args.decay=0
    args.acc_steps=16
    args.max_lr=1e-4
    # args.max_lr=1e-4
    args.pct_start=0.0001
    args.init_lr=1e-6
    args.div_factor=5.0
    args.final_div_factor=1.0
    # GPU and dtype config
    args.multi_gpu=True
    args.zero_optimization=2
    args.use_amp=True
    args.dtype=torch.bfloat16
    args.use_scheduler=True
    # gerneate config
    args.max_new_tokens=512
    args.version=1
    args.model_name='your_model_name'
    args.check_point=None
    args.start_step=0
    args.only_test=False
    # check_point=None
    trainer = Trainer(args) 
    print(args)
    trainer.main()



