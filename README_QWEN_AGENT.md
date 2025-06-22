# Official Qwen-Agent Implementation

This implementation replaces the custom API server with the official [Qwen-Agent framework](https://github.com/QwenLM/Qwen-Agent), providing a more robust and feature-complete solution for multi-model AI assistance.

## 🚀 Key Features

- **Official Qwen-Agent Framework**: Built on the official framework with Function Calling, MCP, Code Interpreter, RAG, and more
- **Dual Model Architecture**: 
  - **Qwen3** (or Qwen2.5-72B) as the main orchestrator with agent capabilities
  - **Qwen2.5-Coder** as the specialized code generation model
- **Smart Task Routing**: Automatically routes coding tasks to the specialized model
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API
- **Web GUI**: Built-in Gradio interface for easy interaction
- **Configurable**: Environment-based configuration with sensible defaults

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Qwen-Agent Framework                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │   Qwen3 Model   │    │        Custom Tools                 │ │
│  │  (Orchestrator) │───▶│  • advanced_code_generator          │ │
│  │  - Function     │    │  • code_interpreter                 │ │
│  │    Calling      │    │  • [Future: web_search, file_ops]  │ │
│  │  - Agent Logic  │    │                                     │ │
│  └─────────────────┘    └─────────────────────────────────────┘ │
│           │                              │                     │
│           │                              ▼                     │
│           │                   ┌─────────────────────────────┐  │
│           │                   │   Qwen2.5-Coder Model      │  │
│           │                   │   (Code Specialist)        │  │
│           │                   │  - High-quality code gen   │  │
│           │                   │  - Multiple languages      │  │
│           │                   │  - Best practices          │  │
│           │                   └─────────────────────────────┘  │
│           │                                                    │
│           ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Response Generation                            │ │
│  │  • Comprehensive explanations                              │ │
│  │  • Code examples with documentation                        │ │
│  │  • Step-by-step breakdowns                                 │ │
│  │  • Best practices and optimizations                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Installation & Setup

### 1. Install Dependencies

```bash
pip install qwen-agent fastapi uvicorn gradio httpx
```

### 2. Configure Your Models

Edit [`config.yaml`](config.yaml) or set environment variables:

```bash
# Orchestrator Model (Qwen3)
export QWEN3_MODEL="areebtechnology2025/Qwen3-30B-Areeb-Lora"
export QWEN3_URL="http://localhost:8000/v1"

# Code Generation Model (Qwen2.5-Coder)
export CODE_MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
export CODE_MODEL_URL="http://localhost:8001/v1/chat/completions"

# API Server
export API_PORT=8002
export GUI_PORT=7860
```

### 3. Start the Services

#### Option A: API Server (for Continue/Roo Code)
```bash
python qwen_agent_server.py
```

#### Option B: GUI Interface
```bash
python qwen_agent_server.py gui
```

#### Option C: Both (separate terminals)
```bash
# Terminal 1: API Server
python qwen_agent_server.py

# Terminal 2: GUI
python qwen_agent_server.py gui
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_qwen_agent.py
```

## 🌐 API Usage

### OpenAI-Compatible Endpoint

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen-agent",
    "messages": [
      {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
    ],
    "temperature": 0.7,
    "stream": false
  }'
```

### Health Check
```bash
curl http://localhost:8002/health
```

## 🎯 How It Works

### 1. **Request Processing**
- User sends a request to the API endpoint
- Qwen-Agent framework receives and processes the request

### 2. **Smart Task Detection**
- The orchestrator (Qwen3) analyzes the request
- If it detects a coding task, it automatically calls the `advanced_code_generator` tool

### 3. **Specialized Code Generation**
- The `advanced_code_generator` tool routes the request to Qwen2.5-Coder
- The specialized model generates high-quality, well-documented code
- Result is returned to the orchestrator

### 4. **Response Enhancement**
- The orchestrator combines the code with comprehensive explanations
- Adds context, examples, and best practices
- Returns the enhanced response to the user

## 📊 Model Comparison

| Model | Role | Strengths | Use Cases |
|-------|------|-----------|-----------|
| **Qwen3-30B-Areeb-Lora** | Orchestrator | Function calling, reasoning, agent logic | Task routing, explanations, general Q&A |
| **Qwen2.5-Coder-32B-Instruct** | Code Specialist | Advanced code generation, 32B parameters | Complex programming, algorithms, enterprise code |

## 🔧 Configuration Options

### Environment Variables

```bash
# Model Configuration
QWEN3_MODEL=areebtechnology2025/Qwen3-30B-Areeb-Lora
QWEN3_URL=http://localhost:8000/v1
QWEN3_API_KEY=EMPTY
QWEN3_TEMPERATURE=0.7
QWEN3_TOP_P=0.8
QWEN3_MAX_TOKENS=4096

CODE_MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct
CODE_MODEL_URL=http://localhost:8001/v1/chat/completions
CODE_MODEL_API_KEY=EMPTY
CODE_MODEL_TIMEOUT=120

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8002
GUI_HOST=0.0.0.0
GUI_PORT=7860
LOG_LEVEL=INFO

# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=150
ENABLE_CACHING=true
```

## 🛠️ Customization

### Adding New Tools

```python
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool('my_custom_tool')
class MyCustomTool(BaseTool):
    description = 'Description of what this tool does'
    parameters = [{
        'name': 'param_name',
        'type': 'string',
        'description': 'Parameter description',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        # Your tool logic here
        return "Tool result"
```

### Modifying System Prompts

Edit the `AGENT_CONFIG['system_message']` in [`qwen_config.py`](qwen_config.py) to customize the agent's behavior.

## 📈 Performance Optimization

### 1. **Concurrent Requests**
```python
# Adjust in qwen_config.py
PERFORMANCE_CONFIG = {
    'max_concurrent_requests': 20,  # Increase for higher throughput
    'request_timeout': 600,         # Increase for complex tasks
}
```

### 2. **Model Parameters**
```python
# For faster responses (less quality)
QWEN3_CONFIG['generate_cfg']['temperature'] = 0.3
QWEN3_CONFIG['generate_cfg']['max_tokens'] = 4096

# For better quality (slower)
QWEN3_CONFIG['generate_cfg']['temperature'] = 0.8
QWEN3_CONFIG['generate_cfg']['max_tokens'] = 8192
```

## 🔒 Security Considerations

### 1. **API Key Validation**
```yaml
# Enable in config.yaml
authentication:
  enabled: true
  api_keys:
    "your-secure-key":
      name: "Client Name"
      user_email: "user@example.com"
      permissions: ["chat", "code_generation"]
      rate_limit_per_hour: 100
      rate_limit_per_day: 1000
```

### 2. **Network Security**
- Use HTTPS in production
- Implement proper firewall rules
- Consider using a reverse proxy (nginx, Apache)

## 🚨 Troubleshooting

### Common Issues

1. **"Agent not initialized" error**
   - Check that your model servers are running
   - Verify URLs in configuration
   - Check network connectivity

2. **"Code generation failed" error**
   - Verify Qwen2.5-Coder server is accessible
   - Check API endpoint URL
   - Verify model is loaded correctly

3. **Slow responses**
   - Use the secure vLLM script (`./start_vllm_secure.sh`)
   - Adjust `max_tokens` in configuration (currently optimized to 4096/2048)
   - Reduce `temperature` for faster sampling
   - Check model server resources and GPU utilization

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python qwen_agent_server.py
```

## 📚 Additional Resources

- [Official Qwen-Agent Documentation](https://github.com/QwenLM/Qwen-Agent)
- [Qwen Model Documentation](https://github.com/QwenLM/Qwen)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://gradio.app/)

## 💡 Recent Enhancements & Performance

### ✅ **Completed Optimizations**
- **2-GPU vLLM Setup**: Dedicated GPU per model for maximum performance
- **Secure Configuration**: Models accessible only through authenticated API
- **Timeout Optimization**: Reduced from 300s to 120s for faster responses
- **Token Limits**: Optimized to 4096/2048 tokens for speed vs quality balance
- **Async HTTP Client**: Improved request handling and timeout management

### 🚀 **Performance Results**
- **Orchestrator**: 4.9 → 20-30 tokens/s (4-6x faster)
- **Coder**: 22.8 → 60-80 tokens/s (3-4x faster)
- **IDE Extensions**: Much improved responsiveness for Continue/RooCode

### 📋 **Future Enhancements**
- [ ] Add web search capabilities
- [ ] Implement file operations tool
- [ ] Add image generation support
- [ ] Implement RAG for document Q&A
- [ ] Add multi-language support
- [ ] Implement conversation memory
- [ ] Add batch processing capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This implementation is based on the official Qwen-Agent framework and follows the same Apache 2.0 license.

---

**Note**: This implementation provides a production-ready solution with the official Qwen-Agent framework, replacing the custom API with a more robust and feature-complete system. 