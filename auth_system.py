#!/usr/bin/env python3
"""
Comprehensive Authentication and Usage Tracking System
Handles API key management, request logging, user statistics, and analytics
"""

import os
import json
import sqlite3
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class APIKey:
    """API Key data structure."""
    key_id: str
    key_hash: str
    name: str
    user_email: str
    permissions: List[str]
    rate_limit_per_hour: int
    rate_limit_per_day: int
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class RequestLog:
    """Request log data structure."""
    log_id: str
    api_key_id: str
    endpoint: str
    method: str
    request_data: Dict[str, Any]
    response_data: Dict[str, Any]
    status_code: int
    processing_time: float
    tokens_used: int
    model_used: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    cost: float = 0.0

@dataclass
class UsageStats:
    """Usage statistics data structure."""
    api_key_id: str
    date: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    total_cost: float
    avg_processing_time: float
    endpoints_used: Dict[str, int]
    models_used: Dict[str, int]

class AuthenticationSystem:
    """Comprehensive authentication and usage tracking system."""
    
    def __init__(self, db_path: str = "auth_system.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        with self.get_db_connection() as conn:
            # API Keys table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    user_email TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    rate_limit_per_hour INTEGER DEFAULT 100,
                    rate_limit_per_day INTEGER DEFAULT 1000,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    metadata TEXT
                )
            """)
            
            # Request logs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    log_id TEXT PRIMARY KEY,
                    api_key_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    request_data TEXT,
                    response_data TEXT,
                    status_code INTEGER NOT NULL,
                    processing_time REAL NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    model_used TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    cost REAL DEFAULT 0.0,
                    FOREIGN KEY (api_key_id) REFERENCES api_keys (key_id)
                )
            """)
            
            # Usage statistics table (daily aggregates)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_stats (
                    api_key_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0,
                    failed_requests INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    avg_processing_time REAL DEFAULT 0.0,
                    endpoints_used TEXT,
                    models_used TEXT,
                    PRIMARY KEY (api_key_id, date),
                    FOREIGN KEY (api_key_id) REFERENCES api_keys (key_id)
                )
            """)
            
            # Rate limiting table (for tracking current usage)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    api_key_id TEXT NOT NULL,
                    window_start TIMESTAMP NOT NULL,
                    window_type TEXT NOT NULL,
                    request_count INTEGER DEFAULT 0,
                    PRIMARY KEY (api_key_id, window_start, window_type),
                    FOREIGN KEY (api_key_id) REFERENCES api_keys (key_id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_api_key ON request_logs(api_key_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_timestamp ON request_logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_stats_date ON usage_stats(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON rate_limits(window_start, window_type)")
            
    @contextmanager
    def get_db_connection(self):
        """Get database connection with proper error handling."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def generate_api_key(self, name: str, user_email: str, permissions: List[str], 
                        rate_limit_per_hour: int = 100, rate_limit_per_day: int = 1000,
                        metadata: Dict[str, Any] = None) -> Tuple[str, str]:
        """Generate a new API key."""
        # Generate secure API key
        api_key = f"qwen-{secrets.token_urlsafe(32)}"
        key_id = secrets.token_hex(16)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Create API key record
        api_key_record = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_email=user_email,
            permissions=permissions,
            rate_limit_per_hour=rate_limit_per_hour,
            rate_limit_per_day=rate_limit_per_day,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store in database
        with self.get_db_connection() as conn:
            conn.execute("""
                INSERT INTO api_keys 
                (key_id, key_hash, name, user_email, permissions, rate_limit_per_hour, 
                 rate_limit_per_day, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                api_key_record.key_id,
                api_key_record.key_hash,
                api_key_record.name,
                api_key_record.user_email,
                json.dumps(api_key_record.permissions),
                api_key_record.rate_limit_per_hour,
                api_key_record.rate_limit_per_day,
                api_key_record.created_at,
                json.dumps(api_key_record.metadata)
            ))
        
        logger.info(f"Generated new API key for {user_email}: {name}")
        return api_key, key_id
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate an API key and return user information."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self.get_db_connection() as conn:
            row = conn.execute("""
                SELECT * FROM api_keys WHERE key_hash = ? AND is_active = 1
            """, (key_hash,)).fetchone()
            
            if not row:
                return None
            
            # Update last used timestamp
            conn.execute("""
                UPDATE api_keys SET last_used_at = ? WHERE key_id = ?
            """, (datetime.now(), row['key_id']))
            
            return APIKey(
                key_id=row['key_id'],
                key_hash=row['key_hash'],
                name=row['name'],
                user_email=row['user_email'],
                permissions=json.loads(row['permissions']),
                rate_limit_per_hour=row['rate_limit_per_hour'],
                rate_limit_per_day=row['rate_limit_per_day'],
                created_at=row['created_at'],
                last_used_at=row['last_used_at'],
                is_active=bool(row['is_active']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
    
    def check_rate_limit(self, api_key_id: str, api_key_info: APIKey) -> Tuple[bool, Dict[str, Any]]:
        """Check if the API key has exceeded rate limits."""
        now = datetime.now()
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        with self.get_db_connection() as conn:
            # Check hourly limit
            hourly_count = conn.execute("""
                SELECT request_count FROM rate_limits 
                WHERE api_key_id = ? AND window_start = ? AND window_type = 'hour'
            """, (api_key_id, hour_start)).fetchone()
            
            hourly_usage = hourly_count['request_count'] if hourly_count else 0
            
            # Check daily limit
            daily_count = conn.execute("""
                SELECT request_count FROM rate_limits 
                WHERE api_key_id = ? AND window_start = ? AND window_type = 'day'
            """, (api_key_id, day_start)).fetchone()
            
            daily_usage = daily_count['request_count'] if daily_count else 0
            
            # Check limits
            hourly_exceeded = hourly_usage >= api_key_info.rate_limit_per_hour
            daily_exceeded = daily_usage >= api_key_info.rate_limit_per_day
            
            rate_limit_info = {
                'hourly_usage': hourly_usage,
                'hourly_limit': api_key_info.rate_limit_per_hour,
                'daily_usage': daily_usage,
                'daily_limit': api_key_info.rate_limit_per_day,
                'hourly_remaining': max(0, api_key_info.rate_limit_per_hour - hourly_usage),
                'daily_remaining': max(0, api_key_info.rate_limit_per_day - daily_usage),
                'reset_hour': (hour_start + timedelta(hours=1)).isoformat(),
                'reset_day': (day_start + timedelta(days=1)).isoformat()
            }
            
            return not (hourly_exceeded or daily_exceeded), rate_limit_info
    
    def increment_rate_limit(self, api_key_id: str):
        """Increment rate limit counters."""
        now = datetime.now()
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        with self.get_db_connection() as conn:
            # Increment hourly counter
            conn.execute("""
                INSERT INTO rate_limits (api_key_id, window_start, window_type, request_count)
                VALUES (?, ?, 'hour', 1)
                ON CONFLICT(api_key_id, window_start, window_type) 
                DO UPDATE SET request_count = request_count + 1
            """, (api_key_id, hour_start))
            
            # Increment daily counter
            conn.execute("""
                INSERT INTO rate_limits (api_key_id, window_start, window_type, request_count)
                VALUES (?, ?, 'day', 1)
                ON CONFLICT(api_key_id, window_start, window_type) 
                DO UPDATE SET request_count = request_count + 1
            """, (api_key_id, day_start))
    
    def log_request(self, api_key_id: str, endpoint: str, method: str, 
                   request_data: Dict[str, Any], response_data: Dict[str, Any],
                   status_code: int, processing_time: float, tokens_used: int = 0,
                   model_used: str = "", ip_address: str = "", user_agent: str = "",
                   cost: float = 0.0):
        """Log a request for analytics and monitoring."""
        log_id = secrets.token_hex(16)
        
        request_log = RequestLog(
            log_id=log_id,
            api_key_id=api_key_id,
            endpoint=endpoint,
            method=method,
            request_data=request_data,
            response_data=response_data,
            status_code=status_code,
            processing_time=processing_time,
            tokens_used=tokens_used,
            model_used=model_used,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            cost=cost
        )
        
        with self.get_db_connection() as conn:
            conn.execute("""
                INSERT INTO request_logs 
                (log_id, api_key_id, endpoint, method, request_data, response_data,
                 status_code, processing_time, tokens_used, model_used, timestamp,
                 ip_address, user_agent, cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_log.log_id,
                request_log.api_key_id,
                request_log.endpoint,
                request_log.method,
                json.dumps(request_log.request_data),
                json.dumps(request_log.response_data),
                request_log.status_code,
                request_log.processing_time,
                request_log.tokens_used,
                request_log.model_used,
                request_log.timestamp,
                request_log.ip_address,
                request_log.user_agent,
                request_log.cost
            ))
        
        # Update daily statistics
        self.update_daily_stats(api_key_id, request_log)
    
    def update_daily_stats(self, api_key_id: str, request_log: RequestLog):
        """Update daily usage statistics."""
        date_str = request_log.timestamp.strftime('%Y-%m-%d')
        
        with self.get_db_connection() as conn:
            # Get current stats
            current_stats = conn.execute("""
                SELECT * FROM usage_stats WHERE api_key_id = ? AND date = ?
            """, (api_key_id, date_str)).fetchone()
            
            if current_stats:
                # Update existing stats
                endpoints_used = json.loads(current_stats['endpoints_used'])
                models_used = json.loads(current_stats['models_used'])
                
                endpoints_used[request_log.endpoint] = endpoints_used.get(request_log.endpoint, 0) + 1
                models_used[request_log.model_used] = models_used.get(request_log.model_used, 0) + 1
                
                total_requests = current_stats['total_requests'] + 1
                successful_requests = current_stats['successful_requests'] + (1 if request_log.status_code < 400 else 0)
                failed_requests = current_stats['failed_requests'] + (1 if request_log.status_code >= 400 else 0)
                total_tokens = current_stats['total_tokens'] + request_log.tokens_used
                total_cost = current_stats['total_cost'] + request_log.cost
                
                # Calculate new average processing time
                avg_processing_time = (
                    (current_stats['avg_processing_time'] * current_stats['total_requests'] + request_log.processing_time) 
                    / total_requests
                )
                
                conn.execute("""
                    UPDATE usage_stats SET
                        total_requests = ?, successful_requests = ?, failed_requests = ?,
                        total_tokens = ?, total_cost = ?, avg_processing_time = ?,
                        endpoints_used = ?, models_used = ?
                    WHERE api_key_id = ? AND date = ?
                """, (
                    total_requests, successful_requests, failed_requests,
                    total_tokens, total_cost, avg_processing_time,
                    json.dumps(endpoints_used), json.dumps(models_used),
                    api_key_id, date_str
                ))
            else:
                # Create new stats record
                endpoints_used = {request_log.endpoint: 1}
                models_used = {request_log.model_used: 1}
                
                conn.execute("""
                    INSERT INTO usage_stats 
                    (api_key_id, date, total_requests, successful_requests, failed_requests,
                     total_tokens, total_cost, avg_processing_time, endpoints_used, models_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    api_key_id, date_str, 1,
                    1 if request_log.status_code < 400 else 0,
                    1 if request_log.status_code >= 400 else 0,
                    request_log.tokens_used, request_log.cost, request_log.processing_time,
                    json.dumps(endpoints_used), json.dumps(models_used)
                ))
    
    def get_user_statistics(self, api_key_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive user statistics."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with self.get_db_connection() as conn:
            # Get API key info
            api_key_info = conn.execute("""
                SELECT name, user_email, created_at, last_used_at FROM api_keys WHERE key_id = ?
            """, (api_key_id,)).fetchone()
            
            if not api_key_info:
                return {}
            
            # Get daily stats
            daily_stats = conn.execute("""
                SELECT * FROM usage_stats 
                WHERE api_key_id = ? AND date >= ? AND date <= ?
                ORDER BY date
            """, (api_key_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))).fetchall()
            
            # Get recent requests
            recent_requests = conn.execute("""
                SELECT endpoint, method, status_code, processing_time, tokens_used, 
                       model_used, timestamp FROM request_logs
                WHERE api_key_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT 100
            """, (api_key_id, start_date)).fetchall()
            
            # Aggregate statistics
            total_requests = sum(row['total_requests'] for row in daily_stats)
            total_tokens = sum(row['total_tokens'] for row in daily_stats)
            total_cost = sum(row['total_cost'] for row in daily_stats)
            avg_processing_time = sum(row['avg_processing_time'] * row['total_requests'] for row in daily_stats) / max(total_requests, 1)
            
            # Aggregate endpoint and model usage
            all_endpoints = {}
            all_models = {}
            for row in daily_stats:
                endpoints = json.loads(row['endpoints_used'])
                models = json.loads(row['models_used'])
                for endpoint, count in endpoints.items():
                    all_endpoints[endpoint] = all_endpoints.get(endpoint, 0) + count
                for model, count in models.items():
                    all_models[model] = all_models.get(model, 0) + count
            
            return {
                'user_info': {
                    'name': api_key_info['name'],
                    'email': api_key_info['user_email'],
                    'created_at': api_key_info['created_at'],
                    'last_used_at': api_key_info['last_used_at']
                },
                'summary': {
                    'total_requests': total_requests,
                    'total_tokens': total_tokens,
                    'total_cost': total_cost,
                    'avg_processing_time': avg_processing_time,
                    'period_days': days
                },
                'daily_stats': [dict(row) for row in daily_stats],
                'endpoint_usage': all_endpoints,
                'model_usage': all_models,
                'recent_requests': [dict(row) for row in recent_requests]
            }
    
    def get_all_users_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for all users."""
        with self.get_db_connection() as conn:
            users = conn.execute("""
                SELECT key_id, name, user_email, created_at, last_used_at, is_active 
                FROM api_keys ORDER BY created_at DESC
            """).fetchall()
            
            result = []
            for user in users:
                stats = self.get_user_statistics(user['key_id'], days=7)  # Last 7 days
                result.append({
                    'key_id': user['key_id'],
                    'name': user['name'],
                    'email': user['user_email'],
                    'created_at': user['created_at'],
                    'last_used_at': user['last_used_at'],
                    'is_active': bool(user['is_active']),
                    'recent_stats': stats.get('summary', {})
                })
            
            return result
    
    def deactivate_api_key(self, api_key_id: str):
        """Deactivate an API key."""
        with self.get_db_connection() as conn:
            conn.execute("""
                UPDATE api_keys SET is_active = 0 WHERE key_id = ?
            """, (api_key_id,))
        
        logger.info(f"Deactivated API key: {api_key_id}")
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old request logs and rate limit data."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.get_db_connection() as conn:
            # Clean up old request logs
            deleted_logs = conn.execute("""
                DELETE FROM request_logs WHERE timestamp < ?
            """, (cutoff_date,)).rowcount
            
            # Clean up old rate limit data
            deleted_limits = conn.execute("""
                DELETE FROM rate_limits WHERE window_start < ?
            """, (cutoff_date,)).rowcount
            
            logger.info(f"Cleaned up {deleted_logs} old request logs and {deleted_limits} old rate limit records")
            return deleted_logs, deleted_limits

# Global authentication system instance
auth_system = AuthenticationSystem() 