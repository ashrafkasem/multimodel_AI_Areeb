# Qwen-Agent Server

A production-ready implementation using the official [Qwen-Agent framework](https://github.com/QwenLM/Qwen-Agent) with dual-model architecture and comprehensive tooling.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
./install.sh
```

### 2. Configure Your Setup
Edit `config.yaml` to customize models, ports, and settings:
```bash
./run.sh config
```

### 3. Start the Server
```bash
# Start API server only
./run.sh api

# Start GUI interface only  
./run.sh gui

# Start both servers
./run.sh both
```

## 📋 Available Commands

```bash
./run.sh api         # Start API server (port 8002)
./run.sh gui         # Start GUI interface (port 7860)
./run.sh both        # Start both servers
./run.sh stop        # Stop all servers
./run.sh restart     # Restart all servers
./run.sh status      # Check system status
./run.sh test        # Run test suite
./run.sh logs        # Show recent logs
./run.sh config      # Edit configuration
./run.sh help        # Show help
```

## 🔧 Configuration

All settings are managed through `config.yaml`:

- **Models**: Configure orchestrator and code generator models
- **Ports**: Set API (8002) and GUI (7860) ports
- **Authentication**: Enable/disable API key authentication
- **Performance**: Adjust timeouts, concurrency, caching

## 🏗️ Architecture

- **Qwen3** (Orchestrator): Handles agent logic, function calling, reasoning
- **Qwen2.5-Coder** (Specialist): Generates high-quality code
- **FastAPI**: OpenAI-compatible API server
- **Gradio**: Web-based GUI interface

## 📡 API Endpoints

- `POST /v1/chat/completions` - Chat with agent (uses orchestrator + tools)
- `POST /v1/completions` - Fast code completion (direct to coder)
- `GET /health` - Health check
- `GET /` - API information

## 🎯 Use Cases

- **Code Generation**: Automatic routing to specialized models
- **Continue/Roo Code**: OpenAI-compatible code completion
- **Interactive Chat**: Web GUI for direct interaction
- **Function Calling**: Built-in tool support

## 📚 Documentation

- `README_QWEN_AGENT.md` - Comprehensive guide
- `SETUP_SUMMARY.md` - Implementation details
- `config.yaml` - Configuration reference

## 🛠️ Development

```bash
# Run tests
./run.sh test

# Check status
./run.sh status

# View logs
./run.sh logs
```

## 🔒 Security

- API key authentication support
- CORS configuration
- Rate limiting capabilities
- Secure code execution environment

---

**Quick Setup**: `./install.sh && ./run.sh both` 