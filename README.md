## LARGO: Latent Adversarial Reflection through Gradient Optimization for Jailbreaking LLMs

Official implementation for LARGO [(arXiv:2505.10838)](https://arxiv.org/abs/2505.10838), accepted at NeurIPS 2025.

This repository contains a lightweight implementation for LLaMA 2, Phi-3, and Qwen2 single‑prompt attacks and a universal multi‑prompt attack.

By using this project, you acknowledge the Safety and Responsible Use section and agree to comply fully.

### Setup
- Python 3.10+
- CUDA

Install dependencies (example):
```bash
pip install torch transformers pandas scikit-learn
```

### Single‑Prompt Attack
Runs an iterative attack for each row in a dataset.

```bash
python main.py single \
  --dataset_path data/advbench.csv \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --model_family llama2 \
  --cache_dir cache \
  --control_length 200 \
  --lr 1e-3 \
  --weight_decay 0.01 \
  --num_steps 20 \
  --max_iterations 15 \
  --retry 3 \
  --output_path data/adv_results.json
```

### Multi‑Prompt (Universal) Training
Optimizes a universal adversarial suffix over multiple prompts.

```bash
python main.py multi \
  --dataset_path data/advbench.csv \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --cache_dir cache \
  --control_length 200 \
  --lr 1e-3 \
  --weight_decay 0.01 \
  --num_epochs 100 \
  --num_steps 25 \
  --batch_size 10 \
  --num_train_examples 10 \
  --num_test_examples 10 \
  --output_path data/universal.json
```

### Safety and Responsible Use
Use this code strictly for red‑teaming and safety research. By running it, you agree to:
- Use only for lawful, ethical research to improve model safety and defenses.
- Do not generate, distribute, or operationalize harmful content; do not facilitate illegal or dangerous activities.
- Comply with laws, licenses, and institutional policies.
- Securely handle outputs and delete sensitive/harmful artifacts after analysis; do not redistribute them.

If you cannot meet these requirements, do not use this repository.