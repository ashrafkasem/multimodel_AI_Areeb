#!/usr/bin/env python3
"""
Official Qwen-Agent Implementation
Replaces the custom API server with the official Qwen-Agent framework
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.gui import WebUI
import json5
import httpx
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress asyncio socket warnings (common in production)
logging.getLogger("asyncio").setLevel(logging.ERROR)

# Import configuration
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

# FastAPI app for API compatibility
app = FastAPI(title="Qwen-Agent API Server")

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

# Global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent
    logger.info("Initializing Qwen-Agent...")
    agent = create_agent()
    logger.info("Qwen-Agent initialized successfully!")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Extract API key from header if needed
        auth_header = http_request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
            # Add API key validation here if needed
        
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
            
            return {
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
            
    except Exception as e:
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
async def completions(request: CompletionRequest, http_request: Request):
    """Fast code completions - calls Qwen2.5-Coder directly for speed."""
    
    try:
        # Extract API key from header if needed
        auth_header = http_request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
            # Add API key validation here if needed
        
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
                
                return {
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
            
    except httpx.RequestError as e:
        logger.error(f"Error connecting to code model: {e}")
        raise HTTPException(status_code=503, detail=f"Code model unavailable: {str(e)}")
    except Exception as e:
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
        "endpoints": {
            "chat": "/v1/chat/completions (uses Qwen3 orchestrator + tools)",
            "completions": "/v1/completions (direct to Qwen2.5-Coder for speed)",
            "health": "/health",
            "gui": "/gui"
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

def run_gui():
    """Run the Gradio GUI interface."""
    global agent
    if agent is None:
        agent = create_agent()
    
    # Launch the Web UI
    ui = WebUI(agent)
    ui.run(
        server_name=GUI_CONFIG['host'], 
        server_port=GUI_CONFIG['port'], 
        share=GUI_CONFIG['share']
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