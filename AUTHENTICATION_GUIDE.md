# 🔐 Authentication & Usage Tracking System

A comprehensive authentication and usage tracking system for your Qwen-Agent server with API key management, request logging, user statistics, and analytics.

## 🚀 Quick Start

### 1. Enable Authentication

Edit `config.yaml`:
```yaml
authentication:
  enabled: true  # Enable authentication
```

### 2. Create Your First API Key

```bash
# Create an API key for a user
python3 api_key_manager.py create \
  --name "John Doe" \
  --email "john@example.com" \
  --hourly-limit 200 \
  --daily-limit 2000
```

### 3. Use the API Key

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer qwen-YOUR_API_KEY_HERE" \
  -d '{
    "model": "qwen-agent",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 📋 Features

### ✅ **API Key Management**
- Secure API key generation with SHA-256 hashing
- User-specific permissions and rate limits
- Key activation/deactivation
- Metadata support for additional user information

### ✅ **Rate Limiting**
- Hourly and daily rate limits per user
- Automatic rate limit enforcement
- Rate limit headers in responses
- Customizable limits per user

### ✅ **Request Logging**
- Complete request/response logging
- Processing time tracking
- Token usage monitoring
- Cost calculation
- IP address and User-Agent tracking

### ✅ **Usage Analytics**
- Daily usage statistics per user
- System-wide analytics
- Endpoint usage tracking
- Model usage statistics
- Cost analysis and reporting

### ✅ **Admin Interface**
- RESTful admin API endpoints
- User management via API
- Real-time statistics
- Data export capabilities

## 🛠️ API Key Management

### Create API Key

```bash
python3 api_key_manager.py create \
  --name "User Name" \
  --email "user@example.com" \
  --permissions "chat,code_generation" \
  --hourly-limit 100 \
  --daily-limit 1000 \
  --metadata '{"department": "engineering", "project": "ai-tools"}'
```

### List All API Keys

```bash
python3 api_key_manager.py list
```

Output:
```
┌─────────────────────┬──────────────┬─────────────────────┬────────┬────────────┬─────────────────────┬──────────────┬─────────────────┐
│ Key ID              │ Name         │ Email               │ Active │ Created    │ Last Used           │ Requests (7d)│ Tokens (7d)     │
├─────────────────────┼──────────────┼─────────────────────┼────────┼────────────┼─────────────────────┼──────────────┼─────────────────┤
│ a1b2c3d4e5f6...     │ John Doe     │ john@example.com    │ ✅     │ 2024-01-15 │ 2024-01-15 14:30   │ 45           │ 12,340          │
│ f6e5d4c3b2a1...     │ Jane Smith   │ jane@example.com    │ ✅     │ 2024-01-14 │ 2024-01-15 09:15   │ 23           │ 8,920           │
└─────────────────────┴──────────────┴─────────────────────┴────────┴────────────┴─────────────────────┴──────────────┴─────────────────┘
```

### View User Statistics

```bash
# Basic stats
python3 api_key_manager.py stats a1b2c3d4e5f6g7h8

# Detailed stats with daily breakdown
python3 api_key_manager.py stats a1b2c3d4e5f6g7h8 --days 30 --detailed
```

### Deactivate API Key

```bash
python3 api_key_manager.py deactivate a1b2c3d4e5f6g7h8
```

### System Statistics

```bash
python3 api_key_manager.py system
```

## 🌐 Admin API Endpoints

### List All Users
```bash
curl http://localhost:8002/admin/users
```

### Get User Statistics
```bash
curl http://localhost:8002/admin/users/a1b2c3d4e5f6g7h8/stats?days=30
```

### Create New API Key
```bash
curl -X POST http://localhost:8002/admin/users/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "New User",
    "email": "newuser@example.com",
    "permissions": ["chat", "code_generation"],
    "hourly_limit": 150,
    "daily_limit": 1500,
    "metadata": {"team": "research"}
  }'
```

### System Statistics
```bash
curl http://localhost:8002/admin/system/stats
```

### Deactivate User
```bash
curl -X POST http://localhost:8002/admin/users/a1b2c3d4e5f6g7h8/deactivate
```

## 📊 Usage Analytics

### Request Logging

Every API request is automatically logged with:
- **Request Data**: Endpoint, method, parameters (truncated for privacy)
- **Response Data**: Status code, content length, processing time
- **Usage Metrics**: Tokens used, model used, cost calculation
- **User Context**: IP address, User-Agent, timestamp
- **Performance**: Processing time, success/failure status

### Daily Statistics

Daily aggregated statistics include:
- Total requests (successful/failed)
- Token usage and costs
- Average processing time
- Endpoint usage breakdown
- Model usage distribution

### Cost Calculation

Automatic cost calculation based on:
- **Chat Completions**: $0.0001 per token
- **Code Completions**: $0.00005 per token
- Customizable pricing models
- Daily/monthly cost tracking

## 🔒 Security Features

### API Key Security
- **SHA-256 Hashing**: API keys are hashed before storage
- **Secure Generation**: Using `secrets` module for cryptographic security
- **No Plain Text Storage**: Original keys are never stored

### Rate Limiting
- **Sliding Window**: Hourly and daily rate limits
- **Per-User Limits**: Individual rate limits per API key
- **Automatic Enforcement**: Requests blocked when limits exceeded
- **Rate Limit Headers**: Client-friendly rate limit information

### Request Validation
- **Authentication Required**: All endpoints require valid API key
- **Permission Checking**: Role-based access control
- **Input Sanitization**: Request data sanitized and truncated for logging

## 📈 Monitoring & Analytics

### Real-Time Monitoring

```bash
# Watch system stats in real-time
watch -n 5 'python3 api_key_manager.py system'

# Monitor specific user
watch -n 10 'python3 api_key_manager.py stats USER_KEY_ID'
```

### Data Export

```bash
# Export all user data
python3 api_key_manager.py export --output all_users.json

# Export specific user data
python3 api_key_manager.py export --key-id USER_KEY_ID --output user_data.json
```

### Database Cleanup

```bash
# Clean up data older than 90 days
python3 api_key_manager.py cleanup --days 90

# Force cleanup without confirmation
python3 api_key_manager.py cleanup --days 90 --force
```

## 🗄️ Database Schema

The system uses SQLite with the following tables:

### `api_keys`
- User API key information and settings
- Permissions and rate limits
- Creation and usage timestamps

### `request_logs`
- Complete request/response logging
- Performance metrics
- User attribution

### `usage_stats`
- Daily aggregated statistics
- Endpoint and model usage
- Cost calculations

### `rate_limits`
- Real-time rate limiting counters
- Sliding window implementation

## ⚙️ Configuration

### Authentication Settings

```yaml
# config.yaml
authentication:
  enabled: true
  api_keys:
    # Legacy keys (for backward compatibility)
    "V2C-8UkDpfeuisiWxMCkf-5cFpY9zvRxy5MoZ47PVLY":
      name: "Continue/Roo Code"
      permissions: ["chat", "code_generation"]
      rate_limit: 100
  default_permissions: ["chat"]
```

### Performance Settings

```yaml
performance:
  max_concurrent_requests: 10
  request_timeout: 300
  enable_caching: true
  cache_size: 100
```

## 🚨 Rate Limiting

### Rate Limit Headers

When rate limits are enforced, responses include:
```
X-RateLimit-Limit-Hour: 100
X-RateLimit-Remaining-Hour: 45
X-RateLimit-Reset-Hour: 2024-01-15T15:00:00
X-RateLimit-Limit-Day: 1000
X-RateLimit-Remaining-Day: 756
X-RateLimit-Reset-Day: 2024-01-16T00:00:00
```

### Rate Limit Responses

When limits are exceeded:
```json
{
  "detail": "Rate limit exceeded. Hourly: 100/100, Daily: 500/1000",
  "status_code": 429
}
```

## 📝 Usage Examples

### Continue/Roo Code Integration

```json
{
  "models": [
    {
      "title": "Qwen-Agent",
      "provider": "openai",
      "model": "qwen-agent",
      "apiBase": "http://localhost:8002/v1",
      "apiKey": "qwen-YOUR_API_KEY_HERE"
    }
  ]
}
```

### Python Client Example

```python
import requests

headers = {
    'Authorization': 'Bearer qwen-YOUR_API_KEY_HERE',
    'Content-Type': 'application/json'
}

data = {
    'model': 'qwen-agent',
    'messages': [
        {'role': 'user', 'content': 'Write a Python function to calculate fibonacci'}
    ]
}

response = requests.post(
    'http://localhost:8002/v1/chat/completions',
    headers=headers,
    json=data
)

print(response.json())
```

## 🔧 Troubleshooting

### Common Issues

**Authentication Disabled**
```bash
# Check if authentication is enabled
curl http://localhost:8002/ | grep authentication
```

**Invalid API Key**
```bash
# Verify API key exists
python3 api_key_manager.py list | grep YOUR_KEY_ID
```

**Rate Limit Issues**
```bash
# Check user's current rate limits
python3 api_key_manager.py stats YOUR_KEY_ID
```

**Database Issues**
```bash
# Check database file permissions
ls -la auth_system.db

# Reset database (WARNING: This deletes all data)
rm auth_system.db
```

### Debug Mode

Enable debug logging in `config.yaml`:
```yaml
logging:
  level: "DEBUG"
```

## 🚀 Production Deployment

### Security Checklist
- [ ] Change default API keys
- [ ] Enable HTTPS/TLS
- [ ] Set appropriate rate limits
- [ ] Configure firewall rules
- [ ] Set up log rotation
- [ ] Backup database regularly
- [ ] Monitor disk usage
- [ ] Set up alerting

### Performance Optimization
- [ ] Optimize database queries
- [ ] Implement connection pooling
- [ ] Set up caching
- [ ] Monitor memory usage
- [ ] Configure load balancing
- [ ] Set up monitoring

### Backup Strategy
```bash
# Backup database
cp auth_system.db auth_system_backup_$(date +%Y%m%d).db

# Export all data
python3 api_key_manager.py export --output backup_$(date +%Y%m%d).json
```

---

## 📞 Support

For issues or questions:
1. Check the logs: `./run.sh logs`
2. Verify configuration: `./run.sh config`
3. Test connectivity: `./run.sh test`
4. Review this documentation

**Happy coding! 🚀** 