#!/usr/bin/env python3
"""
Redis monitoring script for rate limiting visualization.
Shows rate limit keys, their values, and TTL in real-time.
"""

import asyncio
from redis import asyncio as redis
import time
import json
from typing import Dict, List
from datetime import datetime

class RedisMonitor:
    """Monitor Redis rate limiting keys in real-time."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            print(f"✅ Connected to Redis at {self.redis_url}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            return False
    
    async def get_rate_limit_keys(self) -> List[str]:
        """Get all rate limiting keys."""
        try:
            keys = await self.redis_client.keys("ratelimit:*")
            return sorted(keys)
        except Exception as e:
            print(f"❌ Error getting keys: {e}")
            return []
    
    async def get_key_details(self, key: str) -> Dict:
        """Get detailed information about a rate limit key."""
        try:
            # Get current value and TTL
            value = await self.redis_client.get(key)
            ttl = await self.redis_client.ttl(key)
            
            # Parse key structure: ratelimit:<identifier>:<limit>:<endpoint>
            parts = key.split(":")
            if len(parts) >= 4:
                identifier = parts[1]
                limit = parts[2]
                endpoint = ":".join(parts[3:])
            else:
                identifier = "unknown"
                limit = "unknown"
                endpoint = "unknown"
            
            return {
                "key": key,
                "value": int(value) if value else 0,
                "ttl": ttl,
                "identifier": identifier,
                "limit": limit,
                "endpoint": endpoint,
                "remaining": max(0, int(limit) - int(value)) if value and limit != "unknown" else "unknown"
            }
        except Exception as e:
            return {
                "key": key,
                "error": str(e)
            }
    
    def format_key_info(self, details: Dict) -> str:
        """Format key information for display."""
        if "error" in details:
            return f"❌ {details['key']}: {details['error']}"
        
        remaining = details.get("remaining", "unknown")
        ttl = details.get("ttl", -1)
        
        # Color coding based on remaining requests
        if remaining == "unknown":
            status = "❓"
        elif remaining <= 0:
            status = "🔴"  # Rate limited
        elif remaining <= 3:
            status = "🟡"  # Warning
        else:
            status = "🟢"  # Good
        
        # Format TTL
        if ttl == -1:
            ttl_str = "∞"
        elif ttl == -2:
            ttl_str = "expired"
        else:
            ttl_str = f"{ttl}s"
        
        return (f"{status} {details['endpoint']:<20} "
                f"({details['identifier'][:10]:<10}) "
                f"{details['value']}/{details['limit']} "
                f"[{remaining} left] "
                f"TTL: {ttl_str}")
    
    async def monitor_loop(self, interval: float = 2.0):
        """Main monitoring loop."""
        print(f"🔍 Monitoring Redis rate limiting keys (refresh every {interval}s)")
        print("Press Ctrl+C to stop")
        print("-" * 80)
        
        try:
            while True:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")
                
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"📊 Redis Rate Limiting Monitor - {timestamp}")
                print("=" * 80)
                
                # Get all rate limit keys
                keys = await self.get_rate_limit_keys()
                
                if not keys:
                    print("📭 No rate limiting keys found")
                    print("   (Make some API requests to see rate limiting in action)")
                else:
                    print(f"🔑 Found {len(keys)} rate limiting keys:")
                    print()
                    
                    # Get details for each key
                    for key in keys:
                        details = await self.get_key_details(key)
                        formatted = self.format_key_info(details)
                        print(formatted)
                
                # Summary
                print()
                print("-" * 80)
                print("📈 Summary:")
                print(f"   Total keys: {len(keys)}")
                
                # Count by status
                if keys:
                    details_list = [await self.get_key_details(key) for key in keys]
                    rate_limited = sum(1 for d in details_list if d.get("remaining") == 0)
                    warnings = sum(1 for d in details_list if d.get("remaining", 0) <= 3 and d.get("remaining", 0) > 0)
                    healthy = sum(1 for d in details_list if d.get("remaining", 0) > 3)
                    
                    print(f"   🔴 Rate limited: {rate_limited}")
                    print(f"   🟡 Warnings: {warnings}")
                    print(f"   🟢 Healthy: {healthy}")
                
                print(f"   Next refresh in {interval}s...")
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

async def main():
    """Main function."""
    print("🚀 Redis Rate Limiting Monitor")
    print("=" * 40)
    
    # Get Redis URL from environment or use default
    import os
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    monitor = RedisMonitor(redis_url)
    
    if not await monitor.connect():
        return 1
    
    try:
        await monitor.monitor_loop()
    finally:
        await monitor.close()
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main())) 