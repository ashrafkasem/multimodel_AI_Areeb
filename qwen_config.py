#!/usr/bin/env python3
"""
Configuration file for Qwen-Agent implementation
Adjust these settings based on your model deployment
"""

import os
from typing import Dict, Any

# Model Server Configurations
QWEN3_CONFIG = {
    # Main orchestrator model (Qwen3 or Qwen2.5-72B-Instruct)
    'model': os.getenv('QWEN3_MODEL', 'areebtechnology2025/Qwen3-30B-Areeb-Lora'),
    'model_server': os.getenv('QWEN3_URL', 'http://62.169.159.144:8000/v1'),
    'api_key': os.getenv('QWEN3_API_KEY', 'EMPTY'),
    'generate_cfg': {
        'top_p': float(os.getenv('QWEN3_TOP_P', '0.8')),
        'temperature': float(os.getenv('QWEN3_TEMPERATURE', '0.7')),
        'max_tokens': int(os.getenv('QWEN3_MAX_TOKENS', '8192')),
        'fncall_prompt_type': 'nous',  # Recommended for Qwen3
        'max_input_tokens': 58000,  # Adjust based on your model's context length
    }
}

CODE_MODEL_CONFIG = {
    # Specialized code generation model (Qwen2.5-Coder)
    'model': os.getenv('CODE_MODEL_NAME', 'Qwen/Qwen2.5-Coder-32B-Instruct'),
    'url': os.getenv('CODE_MODEL_URL', 'http://62.169.159.144:8001/v1/chat/completions'),
    'api_key': os.getenv('CODE_MODEL_API_KEY', 'EMPTY'),
    'timeout': int(os.getenv('CODE_MODEL_TIMEOUT', '300')),  # 5 minutes
    'max_tokens': 4096,
    'temperature': 0.1,  # Lower for consistent code generation
    'top_p': 0.95
}

# API Server Configuration
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', '8002')),
    'log_level': os.getenv('LOG_LEVEL', 'info'),
    'cors_origins': os.getenv('CORS_ORIGINS', '*').split(','),
}

# GUI Configuration
GUI_CONFIG = {
    'host': os.getenv('GUI_HOST', '0.0.0.0'),
    'port': int(os.getenv('GUI_PORT', '7860')),
    'share': os.getenv('GUI_SHARE', 'false').lower() == 'true',
    'auth': None  # Add authentication if needed
}

# Agent System Configuration
AGENT_CONFIG = {
    'system_message': """You are an advanced AI assistant powered by the official Qwen-Agent framework with access to specialized tools and capabilities.

Core Capabilities:
- Advanced code generation and programming assistance using specialized models
- Technical problem solving and software architecture design
- Code interpretation and execution in secure environments
- File processing, data analysis, and document understanding
- Web research and information gathering
- Multi-model orchestration for complex tasks

Tool Usage Guidelines:
- Use the advanced_code_generator tool for ANY programming, coding, or software development requests
- Use code_interpreter for running, testing, and debugging code
- Always provide comprehensive explanations with your responses
- Break down complex problems into manageable, logical steps
- Ensure all generated code follows industry best practices and security guidelines
- Suggest optimizations, alternatives, and improvements when appropriate

Response Style:
- Be thorough, educational, and provide clear explanations
- Include practical examples and real-world use cases
- Provide implementation details, context, and reasoning
- Offer multiple approaches when applicable
- Maintain a professional yet friendly tone""",

    'available_tools': [
        'advanced_code_generator',  # Custom tool using Qwen2.5-Coder
        'code_interpreter',         # Built-in code execution
        # Add more tools as needed:
        # 'image_gen',              # Image generation
        # 'web_search',             # Web search capabilities
        # 'file_manager',           # File operations
    ],
    
    'default_files': [],  # Add default files to load if needed
}

# Authentication Configuration (if needed)
AUTH_CONFIG = {
    'enabled': os.getenv('AUTH_ENABLED', 'false').lower() == 'true',
    'api_keys': {
        # Add your API keys here for validation
        'V2C-8UkDpfeuisiWxMCkf-5cFpY9zvRxy5MoZ47PVLY': {
            'name': 'Continue/Roo Code',
            'permissions': ['chat', 'code_generation'],
            'rate_limit': 100  # requests per hour
        },
        # Add more API keys as needed
    },
    'default_permissions': ['chat'],
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.getenv('LOG_FILE', None),  # Set to a file path to log to file
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_concurrent_requests': int(os.getenv('MAX_CONCURRENT_REQUESTS', '10')),
    'request_timeout': int(os.getenv('REQUEST_TIMEOUT', '300')),  # 5 minutes
    'enable_caching': os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
    'cache_size': int(os.getenv('CACHE_SIZE', '100')),
}

def validate_config():
    """Validate the configuration settings."""
    errors = []
    
    # Check required URLs
    if not QWEN3_CONFIG['model_server']:
        errors.append("QWEN3_URL is required")
    
    if not CODE_MODEL_CONFIG['url']:
        errors.append("CODE_MODEL_URL is required")
    
    # Check port availability
    if not (1024 <= API_CONFIG['port'] <= 65535):
        errors.append(f"API_PORT must be between 1024 and 65535, got {API_CONFIG['port']}")
    
    if not (1024 <= GUI_CONFIG['port'] <= 65535):
        errors.append(f"GUI_PORT must be between 1024 and 65535, got {GUI_CONFIG['port']}")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")

def print_config():
    """Print current configuration for debugging."""
    print("🔧 Qwen-Agent Configuration")
    print("=" * 40)
    print(f"Orchestrator Model: {QWEN3_CONFIG['model']}")
    print(f"Orchestrator URL: {QWEN3_CONFIG['model_server']}")
    print(f"Code Model: {CODE_MODEL_CONFIG['model']}")
    print(f"Code Model URL: {CODE_MODEL_CONFIG['url']}")
    print(f"API Server: {API_CONFIG['host']}:{API_CONFIG['port']}")
    print(f"GUI Server: {GUI_CONFIG['host']}:{GUI_CONFIG['port']}")
    print(f"Authentication: {'Enabled' if AUTH_CONFIG['enabled'] else 'Disabled'}")
    print("=" * 40)

if __name__ == "__main__":
    validate_config()
    print_config() 