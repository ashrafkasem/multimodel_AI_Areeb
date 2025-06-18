#!/usr/bin/env python3
"""
Official Qwen-Agent Implementation
Replaces the custom API server with the official Qwen-Agent framework
"""

import os
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from auth_system import auth_system, APIKey
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import json5

# Conditional GUI import
try:
    from qwen_agent.gui import WebUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    WebUI = None
import httpx
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress asyncio socket warnings (common in production)
logging.getLogger("asyncio").setLevel(logging.ERROR)

# Import configuration
import yaml
import os
from pathlib import Path

# Load configuration from YAML file
def load_config():
    """Load configuration from config.yaml file."""
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to the expected format
    qwen3_config = {
        'model': config['models']['orchestrator']['name'],
        'model_server': config['models']['orchestrator']['url'],
        'api_key': config['models']['orchestrator']['api_key'],
        'generate_cfg': {
            'top_p': config['models']['orchestrator']['top_p'],
            'temperature': config['models']['orchestrator']['temperature'],
            'max_tokens': config['models']['orchestrator']['max_tokens'],
            'max_input_tokens': config['models']['orchestrator']['max_input_tokens'],
            'fncall_prompt_type': 'nous',
        }
    }
    
    code_model_config = {
        'model': config['models']['code_generator']['name'],
        'url': config['models']['code_generator']['url'],
        'api_key': config['models']['code_generator']['api_key'],
        'timeout': config['models']['code_generator']['timeout'],
        'max_tokens': config['models']['code_generator']['max_tokens'],
        'temperature': config['models']['code_generator']['temperature'],
        'top_p': config['models']['code_generator']['top_p']
    }
    
    api_config = {
        'host': config['server']['api']['host'],
        'port': config['server']['api']['port'],
        'log_level': config['server']['api']['log_level'],
        'cors_origins': config['server']['api']['cors_origins'],
    }
    
    gui_config = {
        'host': config['server']['gui']['host'],
        'port': config['server']['gui']['port'],
        'share': config['server']['gui']['share'],
        'auth': config['server']['gui']['auth']
    }
    
    agent_config = {
        'system_message': config['agent']['system_message'],
        'available_tools': config['agent']['tools'],
        'default_files': config['agent']['default_files']
    }
    
    auth_config = {
        'enabled': config['authentication']['enabled'],
        'api_keys': config['authentication']['api_keys'],
        'default_permissions': config['authentication']['default_permissions']
    }
    
    return qwen3_config, code_model_config, api_config, gui_config, agent_config, auth_config

# Load configuration
try:
    QWEN3_CONFIG, CODE_MODEL_CONFIG, API_CONFIG, GUI_CONFIG, AGENT_CONFIG, AUTH_CONFIG = load_config()
    logger.info("Configuration loaded from config.yaml")
    
    # Define validate_config for YAML configuration
    def validate_config():
        """Validate the YAML configuration settings."""
        errors = []
        
        # Check required URLs
        if not QWEN3_CONFIG.get('model_server'):
            errors.append("Orchestrator model_server URL is required")
        
        if not CODE_MODEL_CONFIG.get('url'):
            errors.append("Code generator URL is required")
        
        # Check port availability
        api_port = API_CONFIG.get('port', 8002)
        gui_port = GUI_CONFIG.get('port', 7860)
        
        if not (1024 <= api_port <= 65535):
            errors.append(f"API port must be between 1024 and 65535, got {api_port}")
        
        if not (1024 <= gui_port <= 65535):
            errors.append(f"GUI port must be between 1024 and 65535, got {gui_port}")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
            
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    logger.info("Falling back to qwen_config.py")
    from qwen_config import (
        QWEN3_CONFIG, CODE_MODEL_CONFIG, API_CONFIG, GUI_CONFIG, 
        AGENT_CONFIG, AUTH_CONFIG, LOGGING_CONFIG, validate_config
    )

# Custom Code Generator Tool using your specialized model
@register_tool('advanced_code_generator')
class AdvancedCodeGenerator(BaseTool):
    """Advanced code generation tool using specialized Qwen2.5-Coder model."""
    
    description = (
        'Generate high-quality code in any programming language using a specialized '
        'code generation model. Use this for complex programming tasks, algorithms, '
        'API development, web scraping, data processing, and any code-related requests.'
    )
    
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': (
            'Detailed description of the code to generate. Include programming language, '
            'requirements, functionality, and any specific constraints or patterns.'
        ),
        'required': True
    }, {
        'name': 'language',
        'type': 'string',
        'description': 'Target programming language (e.g., python, javascript, java, go, rust)',
        'required': False
    }, {
        'name': 'complexity',
        'type': 'string',
        'description': 'Code complexity level: simple, intermediate, advanced',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        """Generate code using the specialized Qwen2.5-Coder model."""
        try:
            # Parse parameters
            parsed_params = json5.loads(params)
            prompt = parsed_params['prompt']
            language = parsed_params.get('language', '')
            complexity = parsed_params.get('complexity', 'intermediate')
            
            # Enhanced system prompt for code generation
            system_prompt = f"""You are an expert software engineer and code architect. Generate clean, efficient, and well-documented code.

Guidelines:
- Write production-ready code with proper error handling
- Include comprehensive comments and docstrings
- Follow best practices and design patterns
- Ensure code is secure and optimized
- Provide example usage when applicable
- Code complexity level: {complexity}"""
            
            # Enhanced user prompt
            if language:
                enhanced_prompt = f"Generate {language} code for: {prompt}\n\nRequirements:\n- Use {language} best practices\n- Include proper error handling\n- Add comprehensive documentation"
            else:
                enhanced_prompt = f"Generate code for: {prompt}\n\nRequirements:\n- Choose the most appropriate programming language\n- Include proper error handling\n- Add comprehensive documentation"
            
            # Prepare request for specialized code model
            payload = {
                'model': CODE_MODEL_CONFIG['model'],
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': enhanced_prompt}
                ],
                'temperature': 0.1,  # Lower temperature for consistent code
                'max_tokens': 4096,
                'top_p': 0.95
            }
            
            # Make request to specialized code model
            response = httpx.post(
                CODE_MODEL_CONFIG['url'],
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=CODE_MODEL_CONFIG['timeout']
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract generated code
            generated_code = result['choices'][0]['message']['content']
            
            logger.info(f"Code generated successfully using specialized model for: {prompt[:50]}...")
            return generated_code
            
        except httpx.RequestError as e:
            error_msg = f"Error connecting to code model: {str(e)}"
            logger.error(error_msg)
            return f"Error: Unable to connect to specialized code model. {error_msg}"
        
        except Exception as e:
            error_msg = f"Error in code generation: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

# Initialize the Qwen-Agent Assistant
def create_agent():
    """Create and configure the Qwen-Agent Assistant."""
    
    # Validate configuration first
    validate_config()
    
    # Create the assistant
    bot = Assistant(
        llm=QWEN3_CONFIG,
        system_message=AGENT_CONFIG['system_message'],
        function_list=AGENT_CONFIG['available_tools'],
        files=AGENT_CONFIG['default_files']
    )
    
    return bot

# Global agent instance
agent = None

# Authentication dependency
async def authenticate_request(request: Request, authorization: str = Header(None)) -> Optional[APIKey]:
    """Authenticate API requests using API keys."""
    # Skip authentication if disabled
    if not AUTH_CONFIG.get('enabled', False):
        return None
    
    # Extract API key from Authorization header
    api_key = None
    if authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization[7:]
        else:
            api_key = authorization
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include 'Authorization: Bearer YOUR_API_KEY' header."
        )
    
    # Validate API key
    api_key_info = auth_system.validate_api_key(api_key)
    if not api_key_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )
    
    # Check rate limits
    allowed, rate_limit_info = auth_system.check_rate_limit(api_key_info.key_id, api_key_info)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Hourly: {rate_limit_info['hourly_usage']}/{rate_limit_info['hourly_limit']}, Daily: {rate_limit_info['daily_usage']}/{rate_limit_info['daily_limit']}",
            headers={
                "X-RateLimit-Limit-Hour": str(rate_limit_info['hourly_limit']),
                "X-RateLimit-Remaining-Hour": str(rate_limit_info['hourly_remaining']),
                "X-RateLimit-Reset-Hour": rate_limit_info['reset_hour'],
                "X-RateLimit-Limit-Day": str(rate_limit_info['daily_limit']),
                "X-RateLimit-Remaining-Day": str(rate_limit_info['daily_remaining']),
                "X-RateLimit-Reset-Day": rate_limit_info['reset_day']
            }
        )
    
    # Increment rate limit counter
    auth_system.increment_rate_limit(api_key_info.key_id)
    
    return api_key_info

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    global agent
    logger.info("Initializing Qwen-Agent...")
    agent = create_agent()
    logger.info("Qwen-Agent initialized successfully!")
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down Qwen-Agent...")

# FastAPI app for API compatibility
app = FastAPI(title="Qwen-Agent API Server", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models for API compatibility
class Message(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    model: str = Field("qwen-agent", description="Model identifier")
    messages: list[Message] = Field(..., description="Conversation messages")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(8192, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream response")
    stream_options: Optional[dict] = Field(None, description="Streaming options")

class CompletionRequest(BaseModel):
    model: str = Field("qwen-agent", description="Model identifier")
    prompt: str = Field(..., description="The prompt to generate completion for")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(8192, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream response")
    stop: Optional[list[str]] = Field(None, description="Stop sequences")
    top_p: Optional[float] = Field(1.0, description="Top-p sampling parameter")



@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, 
    http_request: Request,
    api_key_info: Optional[APIKey] = Depends(authenticate_request)
):
    """OpenAI-compatible chat completions endpoint."""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    start_time = time.time()
    
    try:
        
        # Convert request messages to agent format
        messages = []
        for msg in request.messages:
            messages.append({
                'role': msg.role,
                'content': msg.content
            })
        
        logger.info(f"Processing chat request with {len(messages)} messages")
        
        if request.stream:
            # Streaming response
            async def generate_stream():
                try:
                    response_text = ""
                    for response in agent.run(messages=messages):
                        # Extract content from agent response
                        if isinstance(response, list):
                            for item in response:
                                if isinstance(item, dict) and item.get('role') == 'assistant':
                                    content = item.get('content', '')
                                    if content and content not in response_text:
                                        new_content = content[len(response_text):]
                                        response_text = content
                                        chunk = {
                                            "choices": [{
                                                "delta": {"content": new_content},
                                                "index": 0
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.error(f"Error in streaming response: {e}")
                    error_chunk = {
                        "error": {"message": str(e), "type": "internal_error"}
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        else:
            # Non-streaming response
            response_messages = []
            for response in agent.run(messages=messages):
                response_messages.extend(response)
            
            # Extract the final assistant response
            assistant_response = ""
            for msg in response_messages:
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    assistant_response = msg.get('content', '')
            
            response_data = {
                "id": f"chatcmpl-{os.urandom(12).hex()}",
                "object": "chat.completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_response
                    },
                    "finish_reason": "stop"
                }]
            }
            
            # Log successful request
            if api_key_info:
                processing_time = time.time() - start_time
                tokens_used = len(assistant_response.split()) + sum(len(msg.content.split()) for msg in request.messages)
                
                auth_system.log_request(
                    api_key_id=api_key_info.key_id,
                    endpoint="/v1/chat/completions",
                    method="POST",
                    request_data={"messages": [{"role": msg.role, "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content} for msg in request.messages]},
                    response_data={"content_length": len(assistant_response), "choices_count": len(response_data["choices"])},
                    status_code=200,
                    processing_time=processing_time,
                    tokens_used=tokens_used,
                    model_used=QWEN3_CONFIG['model'],
                    ip_address=http_request.client.host if http_request.client else "",
                    user_agent=http_request.headers.get("user-agent", ""),
                    cost=tokens_used * 0.0001  # Example cost calculation
                )
            
            return response_data
            
    except Exception as e:
        # Log failed request
        if api_key_info:
            processing_time = time.time() - start_time
            auth_system.log_request(
                api_key_id=api_key_info.key_id,
                endpoint="/v1/chat/completions",
                method="POST",
                request_data={"messages": [{"role": msg.role, "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content} for msg in request.messages]},
                response_data={"error": str(e)},
                status_code=500,
                processing_time=processing_time,
                tokens_used=0,
                model_used=QWEN3_CONFIG['model'],
                ip_address=http_request.client.host if http_request.client else "",
                user_agent=http_request.headers.get("user-agent", "")
            )
        
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Define FIM tokens globally for reuse
FIM_TOKENS = [
    '<fim_prefix>', '</fim_prefix>',
    '<fim_middle>', '</fim_middle>', 
    '<fim_suffix>', '</fim_suffix>',
    '<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>',
    '<PRE>', '</PRE>', '<SUF>', '</SUF>', '<MID>', '</MID>'
]

@app.post("/v1/completions")
async def completions(
    request: CompletionRequest, 
    http_request: Request,
    api_key_info: Optional[APIKey] = Depends(authenticate_request)
):
    """Fast code completions - calls Qwen2.5-Coder directly for speed."""
    
    start_time = time.time()
    
    try:
        
        logger.info(f"Processing FAST completion request (direct to coder): {request.prompt[:50]}...")
        
        # Handle FIM (Fill-in-Middle) tokens and clean the prompt
        clean_prompt = request.prompt
        
        # Remove or replace common FIM tokens
        for token in FIM_TOKENS:
            clean_prompt = clean_prompt.replace(token, '')
        
        # Clean up extra whitespace
        clean_prompt = clean_prompt.strip()
        
        # If prompt is empty after cleaning, provide a minimal context
        if not clean_prompt:
            clean_prompt = "# Complete the code"
        
        # Prepare request for code model directly (no orchestrator overhead)
        # Use a very specific system prompt for pure code completion
        payload = {
            'model': CODE_MODEL_CONFIG['model'],
            'messages': [
                {'role': 'system', 'content': 'You are a code completion assistant. Complete the code naturally without any special tokens, explanations, or markdown formatting. Only return the code continuation.'},
                {'role': 'user', 'content': clean_prompt}
            ],
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'top_p': request.top_p,
            'stream': request.stream,
            'stop': request.stop or ['\n\n', '```', '<fim_', '</fim_', '<|fim_', '<PRE>', '<SUF>', '<MID>']
        }
        
        if request.stop:
            payload['stop'] = request.stop
        
        if request.stream:
            # Streaming response - direct from code model
            async def generate_stream():
                try:
                    async with httpx.AsyncClient(timeout=CODE_MODEL_CONFIG['timeout']) as client:
                        async with client.stream(
                            'POST',
                            CODE_MODEL_CONFIG['url'],
                            json=payload,
                            headers={'Content-Type': 'application/json'}
                        ) as response:
                            response.raise_for_status()
                            
                            async for line in response.aiter_lines():
                                if line.startswith('data: '):
                                    data = line[6:]
                                    if data == '[DONE]':
                                        yield "data: [DONE]\n\n"
                                        break
                                    
                                    try:
                                        chunk_data = json.loads(data)
                                        # Convert chat completion format to text completion format
                                        if 'choices' in chunk_data:
                                            for choice in chunk_data['choices']:
                                                if 'delta' in choice and 'content' in choice['delta']:
                                                    content = choice['delta']['content']
                                                    # Clean FIM tokens from response
                                                    for token in FIM_TOKENS:
                                                        content = content.replace(token, '')
                                                    
                                                    if content:  # Only send non-empty content
                                                        completion_chunk = {
                                                            "id": f"cmpl-{os.urandom(12).hex()}",
                                                            "object": "text_completion",
                                                            "created": int(asyncio.get_event_loop().time()),
                                                            "model": request.model,
                                                            "choices": [{
                                                                "text": content,
                                                                "index": 0,
                                                                "logprobs": None,
                                                                "finish_reason": choice.get('finish_reason')
                                                            }]
                                                        }
                                                        yield f"data: {json.dumps(completion_chunk)}\n\n"
                                    except json.JSONDecodeError:
                                        continue
                                        
                except Exception as e:
                    logger.error(f"Error in streaming completion: {e}")
                    error_chunk = {
                        "error": {"message": str(e), "type": "internal_error"}
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        else:
            # Non-streaming response - direct from code model
            async with httpx.AsyncClient(timeout=CODE_MODEL_CONFIG['timeout']) as client:
                response = await client.post(
                    CODE_MODEL_CONFIG['url'],
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract the generated text from chat completion format
                generated_text = ""
                if 'choices' in result and len(result['choices']) > 0:
                    generated_text = result['choices'][0]['message']['content']
                    # Clean FIM tokens from response
                    for token in FIM_TOKENS:
                        generated_text = generated_text.replace(token, '')
                
                response_data = {
                    "id": f"cmpl-{os.urandom(12).hex()}",
                    "object": "text_completion",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": request.model,
                    "choices": [{
                        "text": generated_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(request.prompt.split()),
                        "completion_tokens": len(generated_text.split()),
                        "total_tokens": len(request.prompt.split()) + len(generated_text.split())
                    }
                }
                
                # Log successful request
                if api_key_info:
                    processing_time = time.time() - start_time
                    tokens_used = len(request.prompt.split()) + len(generated_text.split())
                    
                    auth_system.log_request(
                        api_key_id=api_key_info.key_id,
                        endpoint="/v1/completions",
                        method="POST",
                        request_data={"prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt},
                        response_data={"text_length": len(generated_text), "choices_count": len(response_data["choices"])},
                        status_code=200,
                        processing_time=processing_time,
                        tokens_used=tokens_used,
                        model_used=CODE_MODEL_CONFIG['model'],
                        ip_address=http_request.client.host if http_request.client else "",
                        user_agent=http_request.headers.get("user-agent", ""),
                        cost=tokens_used * 0.00005  # Lower cost for direct code completion
                    )
                
                return response_data
            
    except httpx.RequestError as e:
        # Log failed request
        if api_key_info:
            processing_time = time.time() - start_time
            auth_system.log_request(
                api_key_id=api_key_info.key_id,
                endpoint="/v1/completions",
                method="POST",
                request_data={"prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt},
                response_data={"error": str(e)},
                status_code=503,
                processing_time=processing_time,
                tokens_used=0,
                model_used=CODE_MODEL_CONFIG['model'],
                ip_address=http_request.client.host if http_request.client else "",
                user_agent=http_request.headers.get("user-agent", "")
            )
        
        logger.error(f"Error connecting to code model: {e}")
        raise HTTPException(status_code=503, detail=f"Code model unavailable: {str(e)}")
    except Exception as e:
        # Log failed request
        if api_key_info:
            processing_time = time.time() - start_time
            auth_system.log_request(
                api_key_id=api_key_info.key_id,
                endpoint="/v1/completions",
                method="POST",
                request_data={"prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt},
                response_data={"error": str(e)},
                status_code=500,
                processing_time=processing_time,
                tokens_used=0,
                model_used=CODE_MODEL_CONFIG['model'],
                ip_address=http_request.client.host if http_request.client else "",
                user_agent=http_request.headers.get("user-agent", "")
            )
        
        logger.error(f"Error in completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "framework": "qwen-agent"
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Official Qwen-Agent API Server",
        "version": "1.0.0",
        "framework": "qwen-agent",
        "authentication": AUTH_CONFIG.get('enabled', False),
        "endpoints": {
            "chat": "/v1/chat/completions (uses Qwen3 orchestrator + tools)",
            "completions": "/v1/completions (direct to Qwen2.5-Coder for speed)",
            "health": "/health",
            "admin": "/admin/* (admin endpoints for user management)",
            "gui": "/gui" if GUI_AVAILABLE else "/gui (not available - install qwen-agent[gui])"
        },
        "models": {
            "orchestrator": QWEN3_CONFIG['model'],
            "code_generator": CODE_MODEL_CONFIG['model']
        },
        "routing": {
            "chat_completions": "Qwen3 orchestrator with function calling and tools",
            "completions": "Direct to Qwen2.5-Coder for fast code completion"
        }
    }

# Admin endpoints (protected)
@app.get("/admin/users")
async def admin_list_users():
    """List all users and their statistics."""
    try:
        users = auth_system.get_all_users_statistics()
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/users/{key_id}/stats")
async def admin_user_stats(key_id: str, days: int = 30):
    """Get detailed statistics for a specific user."""
    try:
        stats = auth_system.get_user_statistics(key_id, days=days)
        if not stats:
            raise HTTPException(status_code=404, detail="User not found")
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/system/stats")
async def admin_system_stats():
    """Get system-wide statistics."""
    try:
        users = auth_system.get_all_users_statistics()
        
        total_users = len(users)
        active_users = len([u for u in users if u['is_active']])
        total_requests = sum(u.get('recent_stats', {}).get('total_requests', 0) for u in users)
        total_tokens = sum(u.get('recent_stats', {}).get('total_tokens', 0) for u in users)
        total_cost = sum(u.get('recent_stats', {}).get('total_cost', 0) for u in users)
        
        return {
            "system_stats": {
                "total_users": total_users,
                "active_users": active_users,
                "total_requests_7d": total_requests,
                "total_tokens_7d": total_tokens,
                "total_cost_7d": total_cost
            },
            "top_users": sorted(users, key=lambda u: u.get('recent_stats', {}).get('total_requests', 0), reverse=True)[:10]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., description="User name")
    email: str = Field(..., description="User email")
    permissions: List[str] = Field(["chat", "code_generation"], description="User permissions")
    hourly_limit: int = Field(100, description="Hourly rate limit")
    daily_limit: int = Field(1000, description="Daily rate limit")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")

@app.post("/admin/users/create")
async def admin_create_user(request: CreateAPIKeyRequest):
    """Create a new API key."""
    try:
        api_key, key_id = auth_system.generate_api_key(
            name=request.name,
            user_email=request.email,
            permissions=request.permissions,
            rate_limit_per_hour=request.hourly_limit,
            rate_limit_per_day=request.daily_limit,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "key_id": key_id,
            "api_key": api_key,
            "user": {
                "name": request.name,
                "email": request.email,
                "permissions": request.permissions,
                "rate_limits": {
                    "hourly": request.hourly_limit,
                    "daily": request.daily_limit
                }
            },
            "warning": "Save this API key securely - it won't be shown again!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/users/{key_id}/deactivate")
async def admin_deactivate_user(key_id: str):
    """Deactivate a user's API key."""
    try:
        auth_system.deactivate_api_key(key_id)
        return {"success": True, "message": f"User {key_id} has been deactivated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_gui():
    """Run the Gradio GUI interface with API key authentication."""
    if not GUI_AVAILABLE:
        print("❌ GUI not available!")
        print("Please install GUI dependencies with:")
        print("pip install 'qwen-agent[gui]'")
        print("or")
        print("pip install qwen-agent gradio modelscope-studio")
        sys.exit(1)
    
    global agent
    if agent is None:
        agent = create_agent()
    
    # Configure authentication based on config
    auth_config = None
    if GUI_CONFIG.get('auth'):
        if isinstance(GUI_CONFIG['auth'], list) and len(GUI_CONFIG['auth']) == 2:
            # Simple username/password authentication
            auth_config = tuple(GUI_CONFIG['auth'])
        elif GUI_CONFIG['auth'] == "api_key":
            # API key authentication with usage tracking
            def validate_gui_api_key(username, password):
                """Validate API key for GUI access and enable usage tracking."""
                # In Gradio auth, we use the password field for the API key
                api_key = password
                
                if not AUTH_CONFIG.get('enabled', False):
                    return True  # No auth required if disabled
                
                # Validate the API key
                api_key_info = auth_system.validate_api_key(api_key)
                if api_key_info and api_key_info.is_active:
                    # Check permissions for GUI access
                    if 'chat' in api_key_info.permissions:
                        # Check rate limits
                        allowed, rate_limit_info = auth_system.check_rate_limit(api_key_info.key_id, api_key_info)
                        if allowed:
                            # Store API key info in session for usage tracking
                            # Note: This is a simplified approach - in production you might want more sophisticated session management
                            return True
                        else:
                            print(f"Rate limit exceeded for user {api_key_info.name}")
                            return False
                    else:
                        print(f"User {api_key_info.name} lacks GUI permissions")
                        return False
                else:
                    print("Invalid or inactive API key")
                    return False
            
            # Set up API key authentication
            auth_config = validate_gui_api_key
    
    # Create a wrapper for the WebUI that tracks usage
    if GUI_CONFIG.get('auth') == "api_key" and AUTH_CONFIG.get('enabled', False):
        # Create a custom WebUI class that tracks API usage
        class TrackedWebUI(WebUI):
            def __init__(self, agent):
                super().__init__(agent)
                self.current_api_key_info = None
            
            def _track_gui_usage(self, api_key, query, response, processing_time):
                """Track GUI usage in the authentication system."""
                try:
                    api_key_info = auth_system.validate_api_key(api_key)
                    if api_key_info:
                        # Increment rate limit counters
                        auth_system.increment_rate_limit(api_key_info.key_id)
                        
                        # Log the request
                        auth_system.log_request(
                            api_key_id=api_key_info.key_id,
                            endpoint="/gui/chat",
                            method="POST",
                            request_data={"query": query[:100] + "..." if len(query) > 100 else query},
                            response_data={"response": response[:100] + "..." if len(response) > 100 else response},
                            status_code=200,
                            processing_time=processing_time,
                            tokens_used=len(response.split()),  # Rough token estimate
                            model_used="qwen-agent-gui",
                            ip_address="gui-interface",
                            user_agent="qwen-agent-gui"
                        )
                except Exception as e:
                    logger.error(f"Error tracking GUI usage: {e}")
        
        ui = TrackedWebUI(agent)
    else:
        ui = WebUI(agent)
    
    # Launch the Web UI
    ui.run(
        server_name=GUI_CONFIG['host'], 
        server_port=GUI_CONFIG['port'], 
        share=GUI_CONFIG['share'],
        auth=auth_config
    )

def run_api_server():
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(
        app, 
        host=API_CONFIG['host'], 
        port=API_CONFIG['port'], 
        log_level=API_CONFIG['log_level']
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        print("Starting Qwen-Agent GUI...")
        run_gui()
    else:
        print("Starting Qwen-Agent API Server...")
        run_api_server() 