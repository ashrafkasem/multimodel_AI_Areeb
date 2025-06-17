#!/bin/bash

# Qwen-Agent Start Script
# This script helps you easily start the Qwen-Agent server

set -e

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

# Function to check if a service is running
check_service() {
    local url=$1
    local name=$2
    
    if curl -s --connect-timeout 5 "$url" > /dev/null 2>&1; then
        print_success "$name is running ✓"
        return 0
    else
        print_error "$name is not accessible at $url ✗"
        return 1
    fi
}

# Function to validate configuration
validate_config() {
    print_status "Validating configuration..."
    
    python -c "
import sys
sys.path.append('.')
try:
    from qwen_config import validate_config, print_config
    validate_config()
    print_config()
    print('Configuration validation successful!')
except Exception as e:
    print(f'Configuration error: {e}')
    sys.exit(1)
"
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_deps=()
    
    # Check each dependency individually
    if ! python -c "import qwen_agent" 2>/dev/null; then
        missing_deps+=("qwen-agent")
    fi
    
    if ! python -c "import fastapi" 2>/dev/null; then
        missing_deps+=("fastapi")
    fi
    
    if ! python -c "import uvicorn" 2>/dev/null; then
        missing_deps+=("uvicorn")
    fi
    
    if ! python -c "import gradio" 2>/dev/null; then
        missing_deps+=("gradio")
    fi
    
    if ! python -c "import httpx" 2>/dev/null; then
        missing_deps+=("httpx")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_status "Installing missing dependencies..."
        pip install "${missing_deps[@]}"
    else
        print_success "All dependencies are installed ✓"
    fi
}

# Function to check model servers
check_model_servers() {
    print_status "Checking model servers..."
    
    local qwen3_url="${QWEN3_URL:-http://62.169.159.144:8000/v1}"
    local code_url="${CODE_MODEL_URL:-http://62.169.159.144:8001/v1/chat/completions}"
    
    local servers_ok=true
    
    if ! check_service "$qwen3_url" "Qwen3 Orchestrator"; then
        servers_ok=false
    fi
    
    if ! check_service "$code_url" "Qwen2.5-Coder"; then
        servers_ok=false
    fi
    
    if [ "$servers_ok" = false ]; then
        print_warning "Some model servers are not accessible. The agent may not work correctly."
        print_status "Make sure your model servers are running:"
        print_status "  - Qwen3: $qwen3_url"
        print_status "  - Qwen2.5-Coder: $code_url"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to start the API server
start_api() {
    print_status "Starting Qwen-Agent API server..."
    print_status "The server will be available at: http://localhost:${API_PORT:-8002}"
    print_status "Health check: http://localhost:${API_PORT:-8002}/health"
    print_status "OpenAI-compatible endpoint: http://localhost:${API_PORT:-8002}/v1/chat/completions"
    echo
    print_status "Press Ctrl+C to stop the server"
    echo
    
    python qwen_agent_server.py
}

# Function to start the GUI
start_gui() {
    print_status "Starting Qwen-Agent GUI..."
    print_status "The GUI will be available at: http://localhost:${GUI_PORT:-7860}"
    echo
    print_status "Press Ctrl+C to stop the GUI"
    echo
    
    python qwen_agent_server.py gui
}

# Function to run tests
run_tests() {
    print_status "Running Qwen-Agent tests..."
    python test_qwen_agent.py
}

# Function to show usage
show_usage() {
    echo "Qwen-Agent Start Script"
    echo "======================="
    echo
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  api        Start the API server (default)"
    echo "  gui        Start the GUI interface"
    echo "  test       Run tests"
    echo "  check      Check configuration and dependencies"
    echo "  help       Show this help message"
    echo
    echo "Environment Variables:"
    echo "  QWEN3_URL              Qwen3 orchestrator URL"
    echo "  CODE_MODEL_URL         Qwen2.5-Coder URL"
    echo "  API_PORT               API server port (default: 8002)"
    echo "  GUI_PORT               GUI server port (default: 7860)"
    echo
    echo "Examples:"
    echo "  $0                     # Start API server"
    echo "  $0 gui                 # Start GUI interface"
    echo "  $0 test                # Run tests"
    echo "  API_PORT=8003 $0 api   # Start API on port 8003"
}

# Main script logic
main() {
    local command=${1:-api}
    
    case $command in
        "api")
            check_dependencies
            validate_config
            check_model_servers
            start_api
            ;;
        "gui")
            check_dependencies
            validate_config
            check_model_servers
            start_gui
            ;;
        "test")
            check_dependencies
            validate_config
            run_tests
            ;;
        "check")
            check_dependencies
            validate_config
            check_model_servers
            print_success "All checks passed! ✓"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Banner
echo "🤖 Qwen-Agent Official Implementation"
echo "====================================="
echo

# Run main function
main "$@" 