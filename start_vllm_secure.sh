#!/bin/bash

echo "🔒 Secure vLLM Startup Script"
echo "Starting vLLM models bound to localhost only for security"
echo "🔐 SECURITY: Models accessible only from localhost (127.0.0.1)"
echo

# Ensure vllm command is available
export PATH="$HOME/.local/bin:$PATH"
hash -r  # Clear bash command cache

# Check if vllm command is available
if ! command -v vllm >/dev/null 2>&1; then
    echo "❌ ERROR: vllm command not found!"
    echo "💡 Solutions:"
    echo "   1. Run: source ~/.bashrc"
    echo "   2. Or restart your terminal session"
    echo "   3. Or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
    exit 1
fi

echo "✅ vllm command found: $(which vllm)"

# Check Hugging Face authentication
echo "🔐 Checking Hugging Face authentication..."
if ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
    echo "❌ Not authenticated with Hugging Face!"
    echo "🔑 Model downloads will fail without authentication."
    echo ""
    echo "📋 To authenticate now:"
    echo "1. 🌐 Visit: https://huggingface.co/settings/tokens"
    echo "2. 🔑 Create/copy your token (needs 'Read' permissions)"
    echo "3. 🔐 Run: huggingface-cli login"
    echo ""
    echo "❓ Do you want to authenticate now? (y/n)"
    read -r auth_choice
    
    if [[ "$auth_choice" =~ ^[Yy]$ ]]; then
        echo "🔐 Starting authentication..."
        echo "📝 Paste your token when prompted:"
        huggingface-cli login
        
        # Verify authentication worked
        if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
            echo "✅ Authentication successful!"
        else
            echo "❌ Authentication failed!"
            echo "⚠️  Continuing anyway - some models may fail to download..."
        fi
    else
        echo "⚠️  Skipping authentication - some models may fail to download..."
        echo "💡 You can authenticate later with: huggingface-cli login"
    fi
    echo ""
    sleep 2
else
    echo "✅ Hugging Face authentication verified"
fi

# Set HuggingFace cache directory
export HF_HOME=/ephemeral/

# Kill existing vLLM processes and free GPU memory
echo "Stopping existing vLLM processes..."
pkill -f "vllm serve"
sleep 5

# Force kill if still running
echo "Force killing any remaining vLLM processes..."
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 5

# Clear GPU memory
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 5

# Check GPU memory status
echo "GPU memory status:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits

echo
echo "🚀 Starting vLLM models..."

# Start Orchestrator on GPU 0 (Qwen3-30B-Areeb-Lora)
echo "🔒 Starting Orchestrator on GPU 0 (localhost only)..."
CUDA_VISIBLE_DEVICES=0 HF_HOME=/ephemeral/ nohup vllm serve areebtechnology2025/Qwen3-30B-Areeb-Lora \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  > orchestrator.log 2>&1 &

echo "🔒 Orchestrator starting on GPU 0 (127.0.0.1:8000)... (check orchestrator.log)"
sleep 15

# Start Coder on GPU 1 (Qwen2.5-Coder-32B-Instruct)  
echo "🔒 Starting Coder on GPU 1 (localhost only)..."
CUDA_VISIBLE_DEVICES=1 HF_HOME=/ephemeral/ nohup vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --trust-remote-code \
  > coder.log 2>&1 &

echo "🔒 Coder starting on GPU 1 (127.0.0.1:8001)... (check coder.log)"
sleep 15

echo
echo "✅ Secure vLLM servers started!"
echo "🔒 SECURITY STATUS:"
echo "   • Orchestrator: SECURE (127.0.0.1:8000 - localhost only)"
echo "   • Coder: SECURE (127.0.0.1:8001 - localhost only)"
echo "   • Master API: Ready for startup on 0.0.0.0:8002"
echo
echo "🔐 Security Features:"
echo "   • vLLM models only accessible from localhost"
echo "   • External access blocked for direct model access"
echo "   • No quantization or capacity limits applied"
echo "   • Full model performance and capabilities"
echo
echo "📝 Next Steps:"
echo "   1. Start your master API: ./run.sh api"
echo "   2. Test security: ./test_security.sh"
echo "   3. Test functionality: curl http://localhost:8002/health" 