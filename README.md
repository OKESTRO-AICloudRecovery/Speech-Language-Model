# 🚀 Cloud Failure-Resilient AI Assistant — Multimodal Training & Evaluation Suite
### (FastSLM / FastALM Speech–Language Model Pipeline)

This repository contains the training and evaluation pipeline for a multimodal AI assistant designed to automatically diagnose, interpret, and respond to cloud failure scenarios using speech + text inputs.

The system is built upon a FastSLM / FastALM architecture, combining:
- A Whisper-based speech encoder
- A Qwen3-based language model (LLM)
- Custom multimodal fusion, downsamplers, and audio-text alignment layers

The training module handles speech-language adaptation, and the evaluation module computes WER-based ASR/AST/QA performance on large-scale benchmarks.

---

# ✨ Features (Training Module + Evaluation Module)

## 🔥 1. Multimodal Speech–Language Training Module
### File: train.py

This module provides a complete training loop for a speech-language model using audio + text.

### Key Features

### ✔ FastALM Multimodal Model Integration
- Combines speech encoder outputs (speech_dim) and LLM embeddings (embed_dim)
- Gradient checkpointing to reduce GPU memory usage
- Supports multimodal fusion and text generation

### ✔ Large-Scale Speech Dataset Loader
Supports:
- VoxPopuli (EN)
- LibriSpeech test-clean
- Fleurs Korean (ko_kr)

Uses custom collator for:
- Audio padding
- Text padding
- Label alignment

### ✔ Robust Training Loop
- AMP (bfloat16)
- Gradient accumulation
- Gradient clipping
- NaN/Inf detection
- Auto recovery from data-loader errors
- Automatic checkpoint saving

### ✔ Performance Monitoring
- Optional periodic WER evaluation
- Logs saved into /Results
- Automatically saves best-performing model

---

## 🎧 2. High-Fidelity Evaluation Module
### File: HF_Evaluation.py

Evaluates a trained FastSLM model on ASR, AST, and speech QA tasks.

### Key Features

### ✔ WER-Based Evaluation
- Uses evaluate.load("wer")
- English normalization → EnglishTextNormalizer
- Korean/other languages → BasicTextNormalizer

### ✔ Sequence Generation (Audio + Text)
Uses:

results = model.generate(
    input_ids,
    audio,
    max_new_tokens,
    top_p=0.01,
    top_k=0
)

### ✔ Batch-Level High-Throughput Decoding
- Default batch size: 32
- Measures decoding latency using CUDA events
- Returns:
  - Generated text
  - Reference text
  - Timing matrix

### ✔ Automatic Logging
Stores:
- Epoch
- Step
- WER
- Train loss
- Example sentences (label & prediction)

---

# 🛠 Installation

pip install torch deepspeed transformers evaluate openai-whisper
pip install prettytable tqdm

Ensure CUDA-enabled PyTorch is installed.

---

# ⚙️ Training Usage

## 1. Configure training parameters

Inside the training script:

args.embed_dim = 2560
args.speech_dim = 1280
args.data_path = ['your_dataset']
args.epoch = 1
args.batch_size = 16
args.max_lr = 1e-4
args.use_amp = True
args.multi_gpu = True
args.zero_optimization = 2
args.model_name = 'your_model_name'

---

## 2. Start Training (Single GPU)

python 클라우드_장애극복_멀티모달_훈련_모듈.py

---

## 3. Start Training (Multi-GPU + DeepSpeed)

deepspeed --num_gpus=4 클라우드_장애극복_멀티모달_훈련_모듈.py

---

# 🧪 Evaluation Usage

Modify the bottom of HF_Evaluation.py:

args.repo_id = 'FastSLM'
args.batch_size = 64
args.dtype = torch.bfloat16
args.max_new_tokens = 512

---

## Run evaluation

python HF_Evaluation.py

The script will:
1. Load VoxPopuli, Fleurs Korean, LibriSpeech
2. Generate text from audio
3. Normalize references & predictions
4. Compute WER
5. Save logs into /Results

---

# 📊 Outputs

## ✔ Model checkpoints
Saved under:

/model_weight/your_model_name_step_vX.pt  
/model_weight/Best_your_model_name_vX.pt  

## ✔ Evaluation results (WER)

Stored in:

/Results/your_model_name_vX.txt

Includes:

Epoch, Iteration, Train Loss, WER  
Label: ...  
Generate: ...  

---


# ⭐ Recommended Hardware

Component | Recommendation
--------- | --------------
GPU | A100 40GB / 80GB
Precision | bfloat16
Batch size | 16–64
Multi-GPU | DeepSpeed ZeRO-2

---

# 📝 License

MIT License

---
