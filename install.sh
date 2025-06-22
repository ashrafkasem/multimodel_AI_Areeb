#!/bin/bash

# Qwen-Agent Complete Installation Script
# Installs all dependencies for API server, GUI, and code interpreter

set -e

echo "🚀 Qwen-Agent Complete Setup"
echo "============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
print_status "Checking Python installation..."
if ! command_exists python3; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python version: $PYTHON_VERSION"

# Check pip
if ! command_exists pip3; then
    print_error "pip3 is required but not installed."
    exit 1
fi

print_success "pip3 is available"

# Upgrade pip first
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install core dependencies
print_status "Installing core Qwen-Agent dependencies..."
python3 -m pip install qwen-agent

# Install all extensions
print_status "Installing Qwen-Agent with all extensions..."
python3 -m pip install "qwen-agent[gui]"
python3 -m pip install "qwen-agent[code_interpreter]"

# Install FastAPI and related dependencies
print_status "Installing API server dependencies..."
python3 -m pip install fastapi uvicorn

# Install vLLM for model serving
print_status "Installing vLLM for model serving..."
python3 -m pip install vllm

# Refresh PATH to make vllm command available immediately
print_status "Refreshing environment to recognize vLLM command..."
export PATH="$HOME/.local/bin:$PATH"
hash -r  # Clear bash command cache

# Install utility dependencies
print_status "Installing utility dependencies..."
python3 -m pip install httpx json5 pydantic requests pyyaml tabulate

# Install optional but useful dependencies
print_status "Installing additional useful packages..."
python3 -m pip install python-dotenv  # For .env file support

print_success "All dependencies installed successfully!"

# Make scripts executable
print_status "Setting up executable permissions..."
chmod +x run.sh 2>/dev/null || true
chmod +x start_qwen_agent.sh 2>/dev/null || true

print_success "Setup complete!"

# Important note for remote server installations
print_warning "IMPORTANT for remote servers:"
echo "If you get 'vllm: command not found' errors, run one of these:"
echo "  • source ~/.bashrc"
echo "  • export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "  • Or start a new terminal session"

echo ""
echo "🎉 Installation Summary:"
echo "======================="
echo "✅ Core qwen-agent framework"
echo "✅ GUI support (Gradio interface)"
echo "✅ Code interpreter capabilities"
echo "✅ vLLM model serving engine"
echo "✅ FastAPI server components"
echo "✅ All utility dependencies"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo "1. Configure your models in config.yaml"
echo "2. Start vLLM models: ./start_vllm_secure.sh"
echo "3. Start API server: ./run.sh api"
echo "4. Start GUI: ./run.sh gui (optional)"
echo "5. Test security: ./test_security.sh"
echo "6. Run tests: ./run.sh test"
echo ""
echo "📚 Documentation:"
echo "================="
echo "• README_QWEN_AGENT.md - Comprehensive guide"
echo "• AUTHENTICATION_GUIDE.md - Security & authentication setup"
echo "• SETUP_SUMMARY.md - Quick reference"
echo "• config.yaml - Configuration file"
echo ""
print_success "Ready to use! 🚀" 