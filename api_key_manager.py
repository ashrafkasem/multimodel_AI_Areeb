#!/usr/bin/env python3
"""
API Key Management CLI Tool
Command-line interface for managing API keys, users, and viewing statistics
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from tabulate import tabulate
from auth_system import AuthenticationSystem, auth_system

def create_api_key(args):
    """Create a new API key."""
    try:
        permissions = args.permissions.split(',') if args.permissions else ['chat', 'code_generation']
        metadata = json.loads(args.metadata) if args.metadata else {}
        
        api_key, key_id = auth_system.generate_api_key(
            name=args.name,
            user_email=args.email,
            permissions=permissions,
            rate_limit_per_hour=args.hourly_limit,
            rate_limit_per_day=args.daily_limit,
            metadata=metadata
        )
        
        print(f"✅ API Key created successfully!")
        print(f"Key ID: {key_id}")
        print(f"API Key: {api_key}")
        print(f"User: {args.name} ({args.email})")
        print(f"Permissions: {', '.join(permissions)}")
        print(f"Rate Limits: {args.hourly_limit}/hour, {args.daily_limit}/day")
        print("\n⚠️  Save this API key securely - it won't be shown again!")
        
    except Exception as e:
        print(f"❌ Error creating API key: {e}")
        sys.exit(1)

def list_api_keys(args):
    """List all API keys."""
    try:
        users = auth_system.get_all_users_statistics()
        
        if not users:
            print("No API keys found.")
            return
        
        # Prepare table data
        table_data = []
        for user in users:
            stats = user.get('recent_stats', {})
            table_data.append([
                user['key_id'][:16] + '...',
                user['name'],
                user['email'],
                '✅' if user['is_active'] else '❌',
                user['created_at'].strftime('%Y-%m-%d') if user['created_at'] else 'N/A',
                user['last_used_at'].strftime('%Y-%m-%d %H:%M') if user['last_used_at'] else 'Never',
                stats.get('total_requests', 0),
                stats.get('total_tokens', 0)
            ])
        
        headers = ['Key ID', 'Name', 'Email', 'Active', 'Created', 'Last Used', 'Requests (7d)', 'Tokens (7d)']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
    except Exception as e:
        print(f"❌ Error listing API keys: {e}")
        sys.exit(1)

def show_user_stats(args):
    """Show detailed statistics for a user."""
    try:
        stats = auth_system.get_user_statistics(args.key_id, days=args.days)
        
        if not stats:
            print(f"❌ No data found for key ID: {args.key_id}")
            return
        
        user_info = stats['user_info']
        summary = stats['summary']
        
        print(f"📊 User Statistics - {user_info['name']}")
        print("=" * 50)
        print(f"Email: {user_info['email']}")
        print(f"Created: {user_info['created_at']}")
        print(f"Last Used: {user_info['last_used_at'] or 'Never'}")
        print(f"Period: Last {args.days} days")
        print()
        
        print("📈 Summary:")
        print(f"  Total Requests: {summary['total_requests']:,}")
        print(f"  Total Tokens: {summary['total_tokens']:,}")
        print(f"  Total Cost: ${summary['total_cost']:.4f}")
        print(f"  Avg Processing Time: {summary['avg_processing_time']:.2f}s")
        print()
        
        # Endpoint usage
        if stats['endpoint_usage']:
            print("🎯 Endpoint Usage:")
            endpoint_data = [[endpoint, count] for endpoint, count in stats['endpoint_usage'].items()]
            print(tabulate(endpoint_data, headers=['Endpoint', 'Requests'], tablefmt='grid'))
            print()
        
        # Model usage
        if stats['model_usage']:
            print("🤖 Model Usage:")
            model_data = [[model, count] for model, count in stats['model_usage'].items()]
            print(tabulate(model_data, headers=['Model', 'Requests'], tablefmt='grid'))
            print()
        
        # Daily breakdown
        if stats['daily_stats'] and args.detailed:
            print("📅 Daily Breakdown:")
            daily_data = []
            for day in stats['daily_stats']:
                daily_data.append([
                    day['date'],
                    day['total_requests'],
                    day['successful_requests'],
                    day['failed_requests'],
                    day['total_tokens'],
                    f"${day['total_cost']:.4f}",
                    f"{day['avg_processing_time']:.2f}s"
                ])
            
            headers = ['Date', 'Total', 'Success', 'Failed', 'Tokens', 'Cost', 'Avg Time']
            print(tabulate(daily_data, headers=headers, tablefmt='grid'))
            print()
        
        # Recent requests
        if stats['recent_requests'] and args.detailed:
            print("🕐 Recent Requests (Last 10):")
            recent_data = []
            for req in stats['recent_requests'][:10]:
                recent_data.append([
                    req['timestamp'].strftime('%m-%d %H:%M'),
                    req['endpoint'],
                    req['method'],
                    req['status_code'],
                    f"{req['processing_time']:.2f}s",
                    req['tokens_used'],
                    req['model_used'][:20] + '...' if len(req['model_used']) > 20 else req['model_used']
                ])
            
            headers = ['Time', 'Endpoint', 'Method', 'Status', 'Time', 'Tokens', 'Model']
            print(tabulate(recent_data, headers=headers, tablefmt='grid'))
        
    except Exception as e:
        print(f"❌ Error showing user stats: {e}")
        sys.exit(1)

def deactivate_key(args):
    """Deactivate an API key."""
    try:
        # Confirm deactivation
        if not args.force:
            confirm = input(f"Are you sure you want to deactivate key {args.key_id}? (y/N): ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return
        
        auth_system.deactivate_api_key(args.key_id)
        print(f"✅ API key {args.key_id} has been deactivated.")
        
    except Exception as e:
        print(f"❌ Error deactivating API key: {e}")
        sys.exit(1)

def system_stats(args):
    """Show system-wide statistics."""
    try:
        users = auth_system.get_all_users_statistics()
        
        if not users:
            print("No users found.")
            return
        
        print("🏢 System Statistics")
        print("=" * 50)
        
        total_users = len(users)
        active_users = len([u for u in users if u['is_active']])
        total_requests = sum(u.get('recent_stats', {}).get('total_requests', 0) for u in users)
        total_tokens = sum(u.get('recent_stats', {}).get('total_tokens', 0) for u in users)
        total_cost = sum(u.get('recent_stats', {}).get('total_cost', 0) for u in users)
        
        print(f"Total Users: {total_users}")
        print(f"Active Users: {active_users}")
        print(f"Total Requests (7d): {total_requests:,}")
        print(f"Total Tokens (7d): {total_tokens:,}")
        print(f"Total Cost (7d): ${total_cost:.4f}")
        print()
        
        # Top users by requests
        top_users = sorted(users, key=lambda u: u.get('recent_stats', {}).get('total_requests', 0), reverse=True)[:10]
        if top_users:
            print("🏆 Top Users by Requests (Last 7 days):")
            top_data = []
            for user in top_users:
                stats = user.get('recent_stats', {})
                if stats.get('total_requests', 0) > 0:
                    top_data.append([
                        user['name'],
                        user['email'],
                        stats.get('total_requests', 0),
                        stats.get('total_tokens', 0),
                        f"${stats.get('total_cost', 0):.4f}"
                    ])
            
            if top_data:
                headers = ['Name', 'Email', 'Requests', 'Tokens', 'Cost']
                print(tabulate(top_data, headers=headers, tablefmt='grid'))
        
    except Exception as e:
        print(f"❌ Error showing system stats: {e}")
        sys.exit(1)

def cleanup_data(args):
    """Clean up old data."""
    try:
        if not args.force:
            confirm = input(f"Are you sure you want to delete data older than {args.days} days? (y/N): ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return
        
        deleted_logs, deleted_limits = auth_system.cleanup_old_data(args.days)
        print(f"✅ Cleaned up {deleted_logs} request logs and {deleted_limits} rate limit records.")
        
    except Exception as e:
        print(f"❌ Error cleaning up data: {e}")
        sys.exit(1)

def export_data(args):
    """Export user data to JSON."""
    try:
        if args.key_id:
            # Export specific user
            stats = auth_system.get_user_statistics(args.key_id, days=args.days)
            if not stats:
                print(f"❌ No data found for key ID: {args.key_id}")
                return
            data = {args.key_id: stats}
        else:
            # Export all users
            users = auth_system.get_all_users_statistics()
            data = {}
            for user in users:
                user_stats = auth_system.get_user_statistics(user['key_id'], days=args.days)
                data[user['key_id']] = user_stats
        
        # Write to file
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Data exported to {args.output}")
        
    except Exception as e:
        print(f"❌ Error exporting data: {e}")
        sys.exit(1)

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Qwen-Agent API Key Management Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create API key
    create_parser = subparsers.add_parser('create', help='Create a new API key')
    create_parser.add_argument('--name', required=True, help='User name')
    create_parser.add_argument('--email', required=True, help='User email')
    create_parser.add_argument('--permissions', default='chat,code_generation', help='Comma-separated permissions')
    create_parser.add_argument('--hourly-limit', type=int, default=100, help='Hourly rate limit')
    create_parser.add_argument('--daily-limit', type=int, default=1000, help='Daily rate limit')
    create_parser.add_argument('--metadata', help='JSON metadata')
    
    # List API keys
    list_parser = subparsers.add_parser('list', help='List all API keys')
    
    # Show user statistics
    stats_parser = subparsers.add_parser('stats', help='Show user statistics')
    stats_parser.add_argument('key_id', help='API key ID')
    stats_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    stats_parser.add_argument('--detailed', action='store_true', help='Show detailed breakdown')
    
    # Deactivate API key
    deactivate_parser = subparsers.add_parser('deactivate', help='Deactivate an API key')
    deactivate_parser.add_argument('key_id', help='API key ID to deactivate')
    deactivate_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    
    # System statistics
    system_parser = subparsers.add_parser('system', help='Show system-wide statistics')
    
    # Cleanup old data
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument('--days', type=int, default=90, help='Keep data newer than N days')
    cleanup_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    
    # Export data
    export_parser = subparsers.add_parser('export', help='Export data to JSON')
    export_parser.add_argument('--key-id', help='Export specific user (optional)')
    export_parser.add_argument('--days', type=int, default=30, help='Number of days to export')
    export_parser.add_argument('--output', default='export.json', help='Output file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate function
    commands = {
        'create': create_api_key,
        'list': list_api_keys,
        'stats': show_user_stats,
        'deactivate': deactivate_key,
        'system': system_stats,
        'cleanup': cleanup_data,
        'export': export_data
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == '__main__':
    main() 