"""
Rate Limiting Infrastructure for DubChain API.

This module provides comprehensive rate limiting across all API protocols
with token bucket algorithm, sliding window, and adaptive rate limiting.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

class RateLimitError(Exception):
    """Rate limit exceeded error."""
    pass

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size: int = 60  # seconds
    adaptive_enabled: bool = True
    strict_mode: bool = False

@dataclass
class ClientStats:
    """Client rate limiting statistics."""
    requests: deque = field(default_factory=lambda: deque())
    tokens: float = 10.0
    last_refill: float = field(default_factory=time.time)
    violations: int = 0
    blocked_until: Optional[float] = None
    adaptive_factor: float = 1.0

class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket."""
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_tokens_available(self) -> float:
        """Get available tokens."""
        self._refill()
        return self.tokens

class SlidingWindow:
    """Sliding window rate limiter."""
    
    def __init__(self, window_size: int, max_requests: int):
        """Initialize sliding window."""
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        # Remove old requests outside window
        while self.requests and self.requests[0] <= now - self.window_size:
            self.requests.popleft()
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def get_requests_in_window(self) -> int:
        """Get number of requests in current window."""
        now = time.time()
        
        # Remove old requests
        while self.requests and self.requests[0] <= now - self.window_size:
            self.requests.popleft()
        
        return len(self.requests)

class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load."""
    
    def __init__(self, base_config: RateLimitConfig):
        """Initialize adaptive rate limiter."""
        self.base_config = base_config
        self.current_config = base_config
        self.load_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)
    
    def update_load(self, cpu_usage: float, memory_usage: float, response_time: float):
        """Update system load metrics."""
        self.load_history.append({
            "cpu": cpu_usage,
            "memory": memory_usage,
            "response_time": response_time,
            "timestamp": time.time()
        })
    
    def get_adaptive_factor(self) -> float:
        """Get adaptive factor based on system load."""
        if not self.load_history:
            return 1.0
        
        recent_loads = list(self.load_history)[-10:]  # Last 10 measurements
        
        avg_cpu = sum(load["cpu"] for load in recent_loads) / len(recent_loads)
        avg_memory = sum(load["memory"] for load in recent_loads) / len(recent_loads)
        avg_response_time = sum(load["response_time"] for load in recent_loads) / len(recent_loads)
        
        # Calculate adaptive factor
        factor = 1.0
        
        # Reduce rate if high CPU usage
        if avg_cpu > 80:
            factor *= 0.5
        elif avg_cpu > 60:
            factor *= 0.7
        
        # Reduce rate if high memory usage
        if avg_memory > 90:
            factor *= 0.3
        elif avg_memory > 70:
            factor *= 0.6
        
        # Reduce rate if slow response times
        if avg_response_time > 1000:  # 1 second
            factor *= 0.4
        elif avg_response_time > 500:  # 500ms
            factor *= 0.7
        
        return max(0.1, min(2.0, factor))  # Clamp between 0.1 and 2.0

class RateLimiter:
    """Main rate limiter with multiple algorithms."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter."""
        self.config = config or RateLimitConfig()
        self.clients: Dict[str, ClientStats] = defaultdict(ClientStats)
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.sliding_windows: Dict[str, SlidingWindow] = {}
        self.adaptive_limiter = AdaptiveRateLimiter(self.config)
        
        # Global rate limiting
        self.global_token_bucket = TokenBucket(
            capacity=self.config.burst_limit * 10,
            refill_rate=self.config.requests_per_minute / 60.0
        )
        
        # Cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    async def check_rate_limit(self, client_id: str, operation: str = "request") -> bool:
        """Check rate limit for client."""
        try:
            # Check if client is blocked
            client_stats = self.clients[client_id]
            if client_stats.blocked_until and time.time() < client_stats.blocked_until:
                raise RateLimitError("Client is temporarily blocked")
            
            # Check global rate limit
            if not self.global_token_bucket.consume():
                raise RateLimitError("Global rate limit exceeded")
            
            # Get adaptive factor
            adaptive_factor = self.adaptive_limiter.get_adaptive_factor()
            
            # Check per-client rate limits
            if not await self._check_client_rate_limit(client_id, adaptive_factor):
                client_stats.violations += 1
                
                # Block client if too many violations
                if client_stats.violations > 5:
                    client_stats.blocked_until = time.time() + 300  # 5 minutes
                
                raise RateLimitError("Rate limit exceeded")
            
            # Reset violations on successful request
            if client_stats.violations > 0:
                client_stats.violations = max(0, client_stats.violations - 1)
            
            return True
            
        except RateLimitError:
            raise
        except Exception as e:
            # Log error but don't block request
            print(f"Rate limiting error: {e}")
            return True
    
    async def _check_client_rate_limit(self, client_id: str, adaptive_factor: float) -> bool:
        """Check client-specific rate limits."""
        client_stats = self.clients[client_id]
        
        # Initialize token bucket if not exists
        if client_id not in self.token_buckets:
            self.token_buckets[client_id] = TokenBucket(
                capacity=int(self.config.burst_limit * adaptive_factor),
                refill_rate=(self.config.requests_per_minute / 60.0) * adaptive_factor
            )
        
        # Initialize sliding window if not exists
        if client_id not in self.sliding_windows:
            self.sliding_windows[client_id] = SlidingWindow(
                window_size=self.config.window_size,
                max_requests=int(self.config.requests_per_minute * adaptive_factor)
            )
        
        # Check token bucket
        token_bucket = self.token_buckets[client_id]
        if not token_bucket.consume():
            return False
        
        # Check sliding window
        sliding_window = self.sliding_windows[client_id]
        if not sliding_window.is_allowed():
            return False
        
        # Update client stats
        client_stats.requests.append(time.time())
        client_stats.adaptive_factor = adaptive_factor
        
        return True
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get client rate limiting statistics."""
        if client_id not in self.clients:
            return {}
        
        client_stats = self.clients[client_id]
        token_bucket = self.token_buckets.get(client_id)
        sliding_window = self.sliding_windows.get(client_id)
        
        return {
            "client_id": client_id,
            "requests_count": len(client_stats.requests),
            "violations": client_stats.violations,
            "blocked_until": client_stats.blocked_until,
            "adaptive_factor": client_stats.adaptive_factor,
            "tokens_available": token_bucket.get_tokens_available() if token_bucket else 0,
            "requests_in_window": sliding_window.get_requests_in_window() if sliding_window else 0,
            "is_blocked": client_stats.blocked_until and time.time() < client_stats.blocked_until
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics."""
        return {
            "total_clients": len(self.clients),
            "active_clients": len([c for c in self.clients.values() if len(c.requests) > 0]),
            "blocked_clients": len([c for c in self.clients.values() if c.blocked_until and time.time() < c.blocked_until]),
            "global_tokens_available": self.global_token_bucket.get_tokens_available(),
            "adaptive_factor": self.adaptive_limiter.get_adaptive_factor(),
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "requests_per_day": self.config.requests_per_day,
                "burst_limit": self.config.burst_limit,
                "window_size": self.config.window_size,
                "adaptive_enabled": self.config.adaptive_enabled,
                "strict_mode": self.config.strict_mode
            }
        }
    
    def update_system_metrics(self, cpu_usage: float, memory_usage: float, response_time: float):
        """Update system metrics for adaptive rate limiting."""
        self.adaptive_limiter.update_load(cpu_usage, memory_usage, response_time)
    
    def reset_client(self, client_id: str):
        """Reset client rate limiting."""
        if client_id in self.clients:
            del self.clients[client_id]
        if client_id in self.token_buckets:
            del self.token_buckets[client_id]
        if client_id in self.sliding_windows:
            del self.sliding_windows[client_id]
    
    def _start_cleanup_task(self):
        """Start cleanup task for expired data."""
        async def cleanup():
            while True:
                try:
                    await asyncio.sleep(300)  # Cleanup every 5 minutes
                    self._cleanup_expired_data()
                except Exception as e:
                    print(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup())
    
    def _cleanup_expired_data(self):
        """Clean up expired client data."""
        current_time = time.time()
        expired_clients = []
        
        for client_id, client_stats in self.clients.items():
            # Remove old requests (older than 1 hour)
            while client_stats.requests and client_stats.requests[0] < current_time - 3600:
                client_stats.requests.popleft()
            
            # Remove clients with no recent activity
            if (not client_stats.requests and 
                not client_stats.blocked_until and 
                current_time - client_stats.last_refill > 3600):
                expired_clients.append(client_id)
        
        # Remove expired clients
        for client_id in expired_clients:
            self.reset_client(client_id)

class RateLimitMiddleware:
    """Rate limiting middleware for different protocols."""
    
    def __init__(self, rate_limiter: RateLimiter):
        """Initialize middleware."""
        self.rate_limiter = rate_limiter
    
    async def process_request(self, client_id: str, operation: str = "request") -> bool:
        """Process request through rate limiting."""
        try:
            return await self.rate_limiter.check_rate_limit(client_id, operation)
        except RateLimitError as e:
            # Log rate limit violation
            print(f"Rate limit exceeded for {client_id}: {e}")
            return False
    
    def get_rate_limit_headers(self, client_id: str) -> Dict[str, str]:
        """Get rate limit headers for HTTP responses."""
        stats = self.rate_limiter.get_client_stats(client_id)
        
        headers = {
            "X-RateLimit-Limit": str(self.rate_limiter.config.requests_per_minute),
            "X-RateLimit-Remaining": str(int(stats.get("tokens_available", 0))),
            "X-RateLimit-Reset": str(int(time.time() + 60)),
            "X-RateLimit-Window": str(self.rate_limiter.config.window_size)
        }
        
        if stats.get("is_blocked"):
            headers["X-RateLimit-Blocked"] = "true"
            headers["X-RateLimit-Blocked-Until"] = str(int(stats.get("blocked_until", 0)))
        
        return headers

# Global rate limiter instance
rate_limiter = RateLimiter()
