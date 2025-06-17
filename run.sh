#!/bin/bash

# Qwen-Agent Unified Runner Script
# Single script to handle all operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
SERVER_FILE="$SCRIPT_DIR/qwen_agent_server.py"
TEST_FILE="$SCRIPT_DIR/test_qwen_agent.py"

# Function to print colored output
print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}🤖 Qwen-Agent Management Script${NC}"
    echo -e "${CYAN}================================${NC}"
}

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

# Function to check if process is running
is_process_running() {
    local process_name="$1"
    pgrep -f "$process_name" > /dev/null 2>&1
}

# Function to stop processes
stop_processes() {
    print_status "Stopping Qwen-Agent processes..."
    
    if is_process_running "qwen_agent_server.py"; then
        pkill -f "python.*qwen_agent_server.py" || true
        print_success "API server stopped"
    fi
    
    if is_process_running "gradio"; then
        pkill -f "gradio" || true
        print_success "GUI server stopped"
    fi
    
    sleep 2
}

# Function to check configuration
check_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        print_status "Creating default configuration..."
        # Config file should already exist, but just in case
        return 1
    fi
    
    if [[ ! -f "$SERVER_FILE" ]]; then
        print_error "Server file not found: $SERVER_FILE"
        return 1
    fi
    
    return 0
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v python3 >/dev/null 2>&1; then
        print_error "Python 3 is required but not installed"
        return 1
    fi
    
    # Check if qwen-agent is installed
    if ! python3 -c "import qwen_agent" 2>/dev/null; then
        print_error "qwen-agent is not installed"
        print_status "Run: ./install.sh to install dependencies"
        return 1
    fi
    
    print_success "Dependencies check passed"
    return 0
}

# Function to show status
show_status() {
    print_header
    echo ""
    print_status "System Status:"
    echo ""
    
    # Check if API server is running
    if is_process_running "qwen_agent_server.py"; then
        print_success "✅ API Server: Running (Port 8002)"
    else
        print_warning "❌ API Server: Not running"
    fi
    
    # Check if GUI is running
    if is_process_running "gradio"; then
        print_success "✅ GUI Server: Running (Port 7860)"
    else
        print_warning "❌ GUI Server: Not running"
    fi
    
    echo ""
    print_status "Configuration: $CONFIG_FILE"
    print_status "Server Script: $SERVER_FILE"
    echo ""
}

# Function to start API server
start_api() {
    print_status "Starting Qwen-Agent API Server..."
    
    if is_process_running "qwen_agent_server.py"; then
        print_warning "API server is already running"
        return 0
    fi
    
    if ! check_dependencies; then
        return 1
    fi
    
    if ! check_config; then
        return 1
    fi
    
    print_status "Launching API server on port 8002..."
    cd "$SCRIPT_DIR"
    python3 "$SERVER_FILE" &
    
    # Wait a moment and check if it started
    sleep 3
    if is_process_running "qwen_agent_server.py"; then
        print_success "API server started successfully!"
        print_status "Access at: http://localhost:8002"
        print_status "Health check: http://localhost:8002/health"
    else
        print_error "Failed to start API server"
        return 1
    fi
}

# Function to start GUI
start_gui() {
    print_status "Starting Qwen-Agent GUI..."
    
    if is_process_running "gradio"; then
        print_warning "GUI server is already running"
        return 0
    fi
    
    if ! check_dependencies; then
        return 1
    fi
    
    if ! check_config; then
        return 1
    fi
    
    print_status "Launching GUI server on port 7860..."
    cd "$SCRIPT_DIR"
    python3 "$SERVER_FILE" gui &
    
    # Wait a moment and check if it started
    sleep 5
    if is_process_running "gradio"; then
        print_success "GUI server started successfully!"
        print_status "Access at: http://localhost:7860"
    else
        print_error "Failed to start GUI server"
        print_status "Make sure GUI dependencies are installed:"
        print_status "pip install 'qwen-agent[gui]'"
        return 1
    fi
}

# Function to start both servers
start_both() {
    print_status "Starting both API and GUI servers..."
    
    start_api
    if [[ $? -eq 0 ]]; then
        sleep 2
        start_gui
    fi
}

# Function to run tests
run_tests() {
    print_status "Running Qwen-Agent tests..."
    
    if ! check_dependencies; then
        return 1
    fi
    
    if [[ ! -f "$TEST_FILE" ]]; then
        print_error "Test file not found: $TEST_FILE"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    python3 "$TEST_FILE"
}

# Function to show logs
show_logs() {
    print_status "Showing recent logs..."
    
    if is_process_running "qwen_agent_server.py"; then
        print_status "API Server process is running"
        # Show recent logs if available
        if [[ -f "qwen_agent.log" ]]; then
            tail -n 20 qwen_agent.log
        else
            print_status "No log file found. Check console output."
        fi
    else
        print_warning "API server is not running"
    fi
}

# Function to show help
show_help() {
    print_header
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  api         Start API server only"
    echo "  gui         Start GUI server only"
    echo "  both        Start both API and GUI servers"
    echo "  stop        Stop all running servers"
    echo "  restart     Restart all servers"
    echo "  status      Show system status"
    echo "  test        Run test suite"
    echo "  logs        Show recent logs"
    echo "  install     Install/update dependencies"
    echo "  config      Edit configuration file"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh api          # Start API server"
    echo "  ./run.sh gui          # Start GUI interface"
    echo "  ./run.sh both         # Start both servers"
    echo "  ./run.sh stop         # Stop all servers"
    echo "  ./run.sh status       # Check status"
    echo ""
    echo "Configuration:"
    echo "  Edit config.yaml to customize models, ports, and settings"
    echo ""
}

# Main script logic
case "${1:-help}" in
    "api")
        print_header
        start_api
        ;;
    "gui")
        print_header
        start_gui
        ;;
    "both")
        print_header
        start_both
        ;;
    "stop")
        print_header
        stop_processes
        ;;
    "restart")
        print_header
        stop_processes
        sleep 2
        start_both
        ;;
    "status")
        show_status
        ;;
    "test")
        print_header
        run_tests
        ;;
    "logs")
        print_header
        show_logs
        ;;
    "install")
        print_header
        print_status "Running installation script..."
        if [[ -f "$SCRIPT_DIR/install.sh" ]]; then
            bash "$SCRIPT_DIR/install.sh"
        else
            print_error "install.sh not found"
        fi
        ;;
    "config")
        print_header
        print_status "Opening configuration file..."
        if command -v nano >/dev/null 2>&1; then
            nano "$CONFIG_FILE"
        elif command -v vim >/dev/null 2>&1; then
            vim "$CONFIG_FILE"
        elif command -v code >/dev/null 2>&1; then
            code "$CONFIG_FILE"
        else
            print_status "Please edit: $CONFIG_FILE"
        fi
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 