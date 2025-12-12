import numpy as np
import torch
import torch.nn as nn
import random
import argparse
import os
import evaluate
from tqdm import tqdm
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import deepspeed
import sys
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, GenerationConfig
from FastSLM import FastSLMConfig, FastSLMForConditionalGeneration  # FastALMForCausalLM

# from modeling.Qwen3_FastSLM_v4 import FastALM
from data_loader.ASR_loader import (
    AMI,
    TEDLIUM,
    SPGISPEECH,
    Earnings,
    GigaSpeech,
    LibriSpeech,
    Common_Voice,
    Fleurs,
    Voxpopuli, 
    SLLMDataCollatorWhithPadding, 
)

import warnings

warnings.filterwarnings('ignore')


from prettytable import PrettyTable

conversational_filler = ['UH', 'UHH', 'UM', 'EH', 'MM', 'HM', 'AH', 'HUH', 'HA', 'ER', 'OOF', 'HEE' , 'ACH', 'EEE', 'EW']
unk_tags = ['<UNK>', '<unk>']
gigaspeech_punctuations = ['<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>']
gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']
non_scoring_words = conversational_filler + unk_tags + gigaspeech_punctuations + gigaspeech_garbage_utterance_tags

def asr_text_post_processing(text):
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace('-', ' ')

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    return ' '.join(remaining_words)

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

# tearcher = whisper.load_model('base.en')
# count_parameters(tearcher)
import torch.distributed as dist

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

generation_config = GenerationConfig.from_pretrained('FastSLM')


class Evaluation:
    def __init__(self,args):
        super(Evaluation,self).__init__()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.tokenizer = self._build_model()

    def seed_everything(self,seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.args.multi_gpu:
            torch.cuda.manual_seed_all(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _build_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.args.repo_id,
            trust_remote_code=True
        )
        model = model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.args.repo_id,trust_remote_code=True)
        return model, tokenizer

    
    def save_eval_log(self,clean_ref,clean_res,dataset_name):
        if dataset_name == 'TED-LIUM':
            dummy_clean_ref = [ref for ref in clean_ref if ref != 'ignore time segment in scoring' and ref.strip()]
            clean_res = [res for ref, res in zip(clean_ref, clean_res) if ref != 'ignore time segment in scoring' and ref.strip()]
            clean_ref = dummy_clean_ref
        elif dataset_name in 'gigaspeech':
            clean_ref = [asr_text_post_processing(r) for r in clean_ref]
            clean_res = [asr_text_post_processing(r) for r in clean_res]
        else:
            dummy_clean_ref = [ref for ref in clean_ref if ref.strip()]
            clean_res = [res for ref, res in zip(clean_ref, clean_res) if ref.strip()]
            clean_ref = dummy_clean_ref


        wer = self.metrics_wer.compute(references=clean_ref, predictions=clean_res)
        cer = self.metrics_cer.compute(references=clean_ref, predictions=clean_res)
        print(f'{dataset_name} WER: {wer * 100:.3f}%')
        print(f'{dataset_name} CER: {cer * 100:.3f}%')
        with open(f"Results/ASR_Evaluation_{self.args.version}.txt", 'a') as f:
            f.write(f'{dataset_name} WER: {wer * 100:.3f}% \n')
            f.write(f'{dataset_name} CER: {cer * 100:.3f}% \n\n')
            # f.write('Example of Generate Text\n')
            # for i in range(5):
            #     f.write('Label: {}\n'.format(clean_ref[i]))
            #     f.write('Generate: {}\n'.format(clean_res[i]))
            f.close()
        
    def get_decoding_result(self,loader,lang):
        if lang == 'en':
            normalizer = EnglishTextNormalizer()
        else:
            normalizer = BasicTextNormalizer()
        refs = []
        res = []
        timings = np.zeros((len(loader),1))
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for rep, b in enumerate(tqdm(loader)):
            audio = b['audio']
            input_ids = b['input_ids'].cuda()
            labels = b["labels"].cuda()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.args.dtype):
                    starter.record()
                    results = self.model.generate(
                        input_ids,
                        audio,
                        max_new_tokens=self.args.max_new_tokens,
                        generation_config=generation_config
                    )
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[rep] = curr_time
                    
            for r in results:
                res.append(normalizer(self.tokenizer.decode(r, skip_special_tokens=True)).strip())
            
            for l in labels:
                l[l == -100] = self.tokenizer.eos_token_id
                ref = self.tokenizer.decode(l,skip_special_tokens=True)
                refs.append(normalizer(ref).strip())
        return res, refs, timings

    def main(self):
        self.seed_everything(seed=0)

        print('Dataset Load')
        voxpopuli_test = Voxpopuli('facebook/voxpopuli','test','en')
        korean_test = Fleurs('google/fleurs/ko_kr','test','ko')
        test_clean = LibriSpeech('sanchit-gandhi/librispeech-data','test.clean')
        test_other = LibriSpeech('sanchit-gandhi/librispeech-data','test.other')
        english_test = Fleurs('google/fleurs/en_us','test','en')
        spgispeech_test = SPGISPEECH('kensho/spgispeech','test')
        ami_test = AMI('edinburghcstr/ami','test')
        tedlium_test = TEDLIUM('LIUM/tedlium','test')
        en_common_test = Common_Voice('fsicoli/common_voice_15_0/en','test','en')
        gigaspeech_test = GigaSpeech('speechcolab/gigaspeech','test')
        earnings_test = Earnings('sanchit-gandhi/earnings22_split','test')
        ko_common_test = Common_Voice('fsicoli/common_voice_15_0/ko','test','ko')
        print('Dataset Load Complete')


        voxpopuli_loader = torch.utils.data.DataLoader(dataset=voxpopuli_test,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        ko_loader = torch.utils.data.DataLoader(dataset=korean_test,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        test_loader = torch.utils.data.DataLoader(dataset=test_clean,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        other_loader = torch.utils.data.DataLoader(dataset=test_other,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        ami_loader = torch.utils.data.DataLoader(dataset=ami_test,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        en_loader = torch.utils.data.DataLoader(dataset=english_test,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        spgispeech_loader = torch.utils.data.DataLoader(dataset=spgispeech_test,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        tedlium_loader = torch.utils.data.DataLoader(dataset=tedlium_test,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        en_common_loader = torch.utils.data.DataLoader(dataset=en_common_test,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        gigaspeech_loader = torch.utils.data.DataLoader(dataset=gigaspeech_test,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        earnings_loader = torch.utils.data.DataLoader(dataset=earnings_test,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())
        ko_common_loader = torch.utils.data.DataLoader(dataset=ko_common_test,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.workers,collate_fn=SLLMDataCollatorWhithPadding())

        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")
        test_data = [
            voxpopuli_loader, ko_loader, test_loader, other_loader, 
            ami_loader, en_loader, spgispeech_loader, tedlium_loader, en_common_loader, gigaspeech_loader, 
            earnings_loader, ko_common_loader
        ]
        
        dataset_names = [
            'voxpopuli','Fleurs-Ko','test-clean','test-other','AMI','Fleurs-En', 
            'SpgiSpeech', 'TED-LIUM','Common_Voice 15-En','Gigaspeech','Earnings-22', 'Common_Voice 15-Ko'
        ]
        
        langs = ['en','en','en','en','en','en','en','en','en','en','en','en']
        print('Start Eval')
        self.model.eval()
        for loader, dataset_name,lang in zip(test_data,dataset_names,langs):
            print(f'{dataset_name} Evaluation')
            clean_res, clean_ref, timings = self.get_decoding_result(loader,lang)
            self.save_eval_log(clean_ref,clean_res,dataset_name)
            print('average decoding time: ', np.mean(np.array(timings[1:])))
            print('standard deviation:', np.std(np.array(timings[1:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='speech modality adaptation')
    args = parser.parse_args()
    # model config
    args.embed_dim=2560
    args.speech_dim=1280    
    args.repo_id = 'FastSLM'
    args.batch_size=64
    args.workers=16
    args.dtype=torch.bfloat16
    args.max_new_tokens=512
    args.version=20
    args.zero_optimization=2
    args.multi_gpu=False
    eval = Evaluation(args) 
    print(args)
    eval.main()



