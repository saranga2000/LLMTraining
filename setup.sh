#!/bin/bash
# LLM Training Lab - One-shot setup for Apple Silicon Mac
set -e

echo "🔧 Setting up LLM Training Lab..."

# Create project directory
mkdir -p llm-training-lab
cd llm-training-lab

# Create folder structure
mkdir -p 01_pretrain/data 02_sft 03_lora eval outputs 04_modular

# Create venv
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets peft trl tokenizers accelerate
pip install mlflow matplotlib numpy sentencepiece

echo "✅ Setup complete. To activate env:"
echo "   cd llm-training-lab && source venv/bin/activate"
echo ""
echo "Then start MLflow and run phases:"
echo "   mlflow ui --host 127.0.0.1 --port 5000 &"
echo "   python 01_pretrain/data.py"
