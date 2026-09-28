"""
Rate limiting implementation using token bucket algorithm for production use.

This module provides rate limiting functionality with per-client tracking,
configurable limits, and comprehensive monitoring capabilities.
"""

import asyncio
import time
from typing import Dict, Optional, List, Any
from collections import defaultdict, deque
import logging
from dataclasses import dataclass
from enum import Enum
import uuid


class RateLimitType(Enum):
    """Rate limit types for different client identification methods."""
    IP_BASED = "ip"
    SESSION_BASED = "session"
    API_KEY_BASED = "api_key"
    COMBINED = "combined"


@dataclass
class RateLimitInfo:
    """Rate limit information for a client."""
    remaining_requests: int
    reset_time: float
    limit_per_minute: int
    window_seconds: int
    burst_allowance: int


@dataclass
class QueuedRequest:
    """Represents a queued request waiting for rate limit availability."""
    request_id: str
    client_id: str
    request_weight: int
    queued_at: float
    future: asyncio.Future
    timeout_handle: Optional[asyncio.TimerHandle] = None


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 100
    burst_allowance: int = 10
    window_seconds: int = 60
    cleanup_interval: int = 300  # 5 minutes
    enable_burst: bool = True
    enable_history_tracking: bool = True
    enable_request_queuing: bool = False  # Enable request queuing for burst handling
    max_queue_size: int = 100  # Maximum number of queued requests per client
    queue_timeout: float = 30.0  # Maximum time a request can wait in queue (seconds)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: float, limit_info: RateLimitInfo):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_info = limit_info


class RateLimiter:
    """
    Token bucket rate limiter for production use.
    
    Features:
    - Per-client rate limiting with configurable limits
    - Token bucket algorithm with burst handling
    - Request history tracking for analytics
    - Automatic cleanup of old client data
    - Support for different client identification methods
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize the rate limiter.
        
        Args:
            config: Rate limiting configuration. If None, uses default values.
        """
        self.config = config or RateLimitConfig()
        
        # Token bucket state per client
        self.tokens: Dict[str, float] = defaultdict(lambda: self.config.requests_per_minute)
        self.last_update: Dict[str, float] = defaultdict(time.time)
        
        # Request history for analytics (if enabled)
        if self.config.enable_history_tracking:
            self.request_history: Dict[str, deque] = defaultdict(
                lambda: deque(maxlen=self.config.requests_per_minute * 2)
            )
        
        # Burst tracking
        if self.config.enable_burst:
            self.burst_tokens: Dict[str, int] = defaultdict(lambda: self.config.burst_allowance)
        
        # Request queuing (if enabled)
        if self.config.enable_request_queuing:
            self.request_queues: Dict[str, deque] = defaultdict(deque)
            self.queue_processing_tasks: Dict[str, asyncio.Task] = {}
            self.queued_requests_count: int = 0
            self.processed_from_queue_count: int = 0
            self.queue_timeouts_count: int = 0
        
        # Rate limiting statistics
        self.total_requests: int = 0
        self.blocked_requests: int = 0
        self.active_clients: set = set()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Cleanup task (will be started when first used)
        self._cleanup_task = None
        
        self.logger.info(
            f"Rate limiter initialized: {self.config.requests_per_minute} req/min, "
            f"burst: {self.config.burst_allowance}, window: {self.config.window_seconds}s, "
            f"queuing: {self.config.enable_request_queuing}"
        )
    
    async def is_allowed(
        self, 
        client_id: str, 
        request_weight: int = 1,
        rate_limit_type: RateLimitType = RateLimitType.IP_BASED,
        enable_queuing: bool = True
    ) -> bool:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Unique identifier for the client
            request_weight: Weight of the request (default: 1)
            rate_limit_type: Type of rate limiting to apply
            enable_queuing: Whether to queue the request if rate limited (default: True)
            
        Returns:
            True if request is allowed, False if rate limited
            
        Raises:
            RateLimitExceeded: When rate limit is exceeded (if configured to raise)
        """
        # Start cleanup task if not already running
        if self._cleanup_task is None:
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            except RuntimeError:
                # No event loop running, skip cleanup task
                pass
        
        current_time = time.time()
        
        # Track statistics
        self.total_requests += 1
        self.active_clients.add(client_id)
        
        # Update token bucket
        self._update_tokens(client_id, current_time)
        
        # Check if tokens available
        if self.tokens[client_id] >= request_weight:
            # Consume tokens
            self.tokens[client_id] -= request_weight
            
            # Track request history
            if self.config.enable_history_tracking:
                self.request_history[client_id].append(current_time)
            
            self.logger.debug(
                f"Request allowed for client {client_id}: "
                f"{self.tokens[client_id]:.1f} tokens remaining"
            )
            return True
        
        # Check burst allowance
        if self.config.enable_burst and self.burst_tokens[client_id] >= request_weight:
            self.burst_tokens[client_id] -= request_weight
            
            # Track request history
            if self.config.enable_history_tracking:
                self.request_history[client_id].append(current_time)
            
            self.logger.debug(
                f"Burst request allowed for client {client_id}: "
                f"{self.burst_tokens[client_id]} burst tokens remaining"
            )
            return True
        
        # If queuing is enabled and requested, queue the request
        if self.config.enable_request_queuing and enable_queuing:
            try:
                await self._queue_request(client_id, request_weight)
                return True  # Request was queued and processed successfully
            except asyncio.TimeoutError:
                # Re-raise timeout errors
                raise
            except Exception as e:
                self.logger.error(f"Failed to queue request for client {client_id}: {e}")
                # Fall through to rate limit exceeded
        
        # Rate limit exceeded
        self.blocked_requests += 1
        reset_time = self.get_reset_time(client_id)
        
        self.logger.warning(
            f"Rate limit exceeded for client {client_id} "
            f"(type: {rate_limit_type.value}): "
            f"{self.tokens[client_id]:.1f} tokens, "
            f"{self.burst_tokens.get(client_id, 0) if self.config.enable_burst else 0} burst tokens, "
            f"reset in {reset_time:.1f}s"
        )
        
        return False
    
    async def _queue_request(
        self, 
        client_id: str, 
        request_weight: int
    ) -> None:
        """
        Queue a request for later processing when rate limit allows.
        
        Args:
            client_id: Unique identifier for the client
            request_weight: Weight of the request
            
        Raises:
            Exception: If queue is full or queuing fails
        """
        # Check if queue is full
        if len(self.request_queues[client_id]) >= self.config.max_queue_size:
            self.logger.warning(
                f"Request queue full for client {client_id} "
                f"(size: {len(self.request_queues[client_id])}/{self.config.max_queue_size})"
            )
            raise Exception(f"Request queue full for client {client_id}")
        
        # Create queued request
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        queued_request = QueuedRequest(
            request_id=request_id,
            client_id=client_id,
            request_weight=request_weight,
            queued_at=time.time(),
            future=future
        )
        
        # Set up timeout
        loop = asyncio.get_event_loop()
        timeout_handle = loop.call_later(
            self.config.queue_timeout,
            self._timeout_queued_request,
            queued_request
        )
        queued_request.timeout_handle = timeout_handle
        
        # Add to queue
        self.request_queues[client_id].append(queued_request)
        self.queued_requests_count += 1
        
        self.logger.debug(
            f"Request queued for client {client_id}: "
            f"request_id={request_id}, weight={request_weight}, "
            f"queue_size={len(self.request_queues[client_id])}"
        )
        
        # Start queue processing task if not already running
        if client_id not in self.queue_processing_tasks or \
           self.queue_processing_tasks[client_id].done():
            self.queue_processing_tasks[client_id] = asyncio.create_task(
                self._process_request_queue(client_id)
            )
        
        # Wait for the request to be processed or timeout
        try:
            await future
        except asyncio.CancelledError:
            self.logger.debug(f"Queued request cancelled: {request_id}")
            raise
    
    def _timeout_queued_request(self, queued_request: QueuedRequest) -> None:
        """
        Handle timeout for a queued request.
        
        Args:
            queued_request: The queued request that timed out
        """
        if not queued_request.future.done():
            self.queue_timeouts_count += 1
            self.logger.warning(
                f"Queued request timed out: "
                f"request_id={queued_request.request_id}, "
                f"client_id={queued_request.client_id}, "
                f"wait_time={time.time() - queued_request.queued_at:.2f}s"
            )
            queued_request.future.set_exception(
                asyncio.TimeoutError(f"Request queue timeout after {self.config.queue_timeout}s")
            )
    
    async def _process_request_queue(self, client_id: str) -> None:
        """
        Process queued requests for a client as tokens become available.
        
        Args:
            client_id: Unique identifier for the client
        """
        self.logger.debug(f"Starting queue processing for client {client_id}")
        
        while self.request_queues[client_id]:
            # Get next request from queue
            queued_request = self.request_queues[client_id][0]
            
            # Update tokens
            current_time = time.time()
            self._update_tokens(client_id, current_time)
            
            # Check if we have enough tokens
            if self.tokens[client_id] >= queued_request.request_weight:
                # Remove from queue
                self.request_queues[client_id].popleft()
                
                # Cancel timeout
                if queued_request.timeout_handle:
                    queued_request.timeout_handle.cancel()
                
                # Consume tokens
                self.tokens[client_id] -= queued_request.request_weight
                
                # Track request history
                if self.config.enable_history_tracking:
                    self.request_history[client_id].append(current_time)
                
                # Mark as processed
                if not queued_request.future.done():
                    self.processed_from_queue_count += 1
                    wait_time = current_time - queued_request.queued_at
                    
                    self.logger.debug(
                        f"Processed queued request: "
                        f"request_id={queued_request.request_id}, "
                        f"client_id={client_id}, "
                        f"wait_time={wait_time:.2f}s"
                    )
                    
                    queued_request.future.set_result(True)
            
            # Check burst tokens if regular tokens not available
            elif self.config.enable_burst and \
                 self.burst_tokens[client_id] >= queued_request.request_weight:
                # Remove from queue
                self.request_queues[client_id].popleft()
                
                # Cancel timeout
                if queued_request.timeout_handle:
                    queued_request.timeout_handle.cancel()
                
                # Consume burst tokens
                self.burst_tokens[client_id] -= queued_request.request_weight
                
                # Track request history
                if self.config.enable_history_tracking:
                    self.request_history[client_id].append(current_time)
                
                # Mark as processed
                if not queued_request.future.done():
                    self.processed_from_queue_count += 1
                    wait_time = current_time - queued_request.queued_at
                    
                    self.logger.debug(
                        f"Processed queued request with burst tokens: "
                        f"request_id={queued_request.request_id}, "
                        f"client_id={client_id}, "
                        f"wait_time={wait_time:.2f}s"
                    )
                    
                    queued_request.future.set_result(True)
            else:
                # Not enough tokens yet, wait a bit
                await asyncio.sleep(0.1)  # Check every 100ms
        
        self.logger.debug(f"Queue processing completed for client {client_id}")
    
    def get_queue_stats(self, client_id: str) -> Dict[str, Any]:
        """
        Get queue statistics for a client.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Dictionary with queue statistics
        """
        if not self.config.enable_request_queuing:
            return {'queuing_enabled': False}
        
        queue_size = len(self.request_queues.get(client_id, []))
        
        # Calculate average wait time for queued requests
        avg_wait_time = 0.0
        if queue_size > 0:
            current_time = time.time()
            wait_times = [
                current_time - req.queued_at 
                for req in self.request_queues[client_id]
            ]
            avg_wait_time = sum(wait_times) / len(wait_times)
        
        return {
            'queuing_enabled': True,
            'queue_size': queue_size,
            'max_queue_size': self.config.max_queue_size,
            'queue_timeout': self.config.queue_timeout,
            'avg_wait_time': avg_wait_time,
            'is_processing': client_id in self.queue_processing_tasks and \
                           not self.queue_processing_tasks[client_id].done()
        }
    
    async def check_rate_limit(
        self, 
        client_id: str, 
        request_weight: int = 1,
        raise_on_exceeded: bool = False
    ) -> RateLimitInfo:
        """
        Check rate limit status for a client without consuming tokens.
        
        Args:
            client_id: Unique identifier for the client
            request_weight: Weight of the request to check
            raise_on_exceeded: Whether to raise exception if limit exceeded
            
        Returns:
            RateLimitInfo with current status
            
        Raises:
            RateLimitExceeded: If raise_on_exceeded=True and limit exceeded
        """
        current_time = time.time()
        self._update_tokens(client_id, current_time)
        
        remaining = int(self.tokens[client_id])
        if self.config.enable_burst:
            remaining += self.burst_tokens[client_id]
        
        reset_time = self.get_reset_time(client_id)
        
        limit_info = RateLimitInfo(
            remaining_requests=remaining,
            reset_time=reset_time,
            limit_per_minute=self.config.requests_per_minute,
            window_seconds=self.config.window_seconds,
            burst_allowance=self.config.burst_allowance if self.config.enable_burst else 0
        )
        
        if raise_on_exceeded and remaining < request_weight:
            raise RateLimitExceeded(
                f"Rate limit exceeded for client: {client_id}",
                reset_time,
                limit_info
            )
        
        return limit_info
    
    def _update_tokens(self, client_id: str, current_time: float) -> None:
        """Update token bucket for client."""
        time_passed = current_time - self.last_update[client_id]
        requests_per_second = self.config.requests_per_minute / 60.0
        
        # Add tokens based on time passed
        self.tokens[client_id] = min(
            self.config.requests_per_minute,
            self.tokens[client_id] + time_passed * requests_per_second
        )
        
        # Update burst tokens if enabled
        if self.config.enable_burst:
            # Burst tokens regenerate slower (every 10 seconds)
            burst_regen_rate = self.config.burst_allowance / 600.0  # 10 minutes to full burst
            self.burst_tokens[client_id] = min(
                self.config.burst_allowance,
                self.burst_tokens[client_id] + time_passed * burst_regen_rate
            )
        
        self.last_update[client_id] = current_time
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        current_time = time.time()
        self._update_tokens(client_id, current_time)
        
        remaining = int(self.tokens[client_id])
        if self.config.enable_burst:
            remaining += self.burst_tokens[client_id]
        
        return remaining
    
    def get_reset_time(self, client_id: str) -> float:
        """Get time until rate limit resets for client."""
        if self.tokens[client_id] >= self.config.requests_per_minute:
            return 0.0
        
        tokens_needed = self.config.requests_per_minute - self.tokens[client_id]
        requests_per_second = self.config.requests_per_minute / 60.0
        
        return tokens_needed / requests_per_second
    
    def get_client_stats(self, client_id: str) -> Dict[str, any]:
        """Get detailed statistics for a client."""
        current_time = time.time()
        self._update_tokens(client_id, current_time)
        
        stats = {
            'client_id': client_id,
            'tokens': self.tokens[client_id],
            'remaining_requests': self.get_remaining_requests(client_id),
            'reset_time': self.get_reset_time(client_id),
            'last_update': self.last_update[client_id],
        }
        
        if self.config.enable_burst:
            stats['burst_tokens'] = self.burst_tokens[client_id]
        
        if self.config.enable_history_tracking and client_id in self.request_history:
            history = self.request_history[client_id]
            recent_requests = [
                req_time for req_time in history 
                if current_time - req_time <= self.config.window_seconds
            ]
            stats['requests_in_window'] = len(recent_requests)
            stats['total_requests'] = len(history)
        
        if self.config.enable_request_queuing:
            stats['queue_stats'] = self.get_queue_stats(client_id)
        
        return stats
    
    def get_global_stats(self) -> Dict[str, any]:
        """Get global rate limiter statistics."""
        stats = {
            'total_requests': self.total_requests,
            'blocked_requests': self.blocked_requests,
            'active_clients': len(self.active_clients),
            'block_rate': self.blocked_requests / max(self.total_requests, 1),
            'config': {
                'requests_per_minute': self.config.requests_per_minute,
                'burst_allowance': self.config.burst_allowance,
                'window_seconds': self.config.window_seconds,
                'enable_burst': self.config.enable_burst,
                'enable_history_tracking': self.config.enable_history_tracking,
                'enable_request_queuing': self.config.enable_request_queuing,
            }
        }
        
        if self.config.enable_request_queuing:
            total_queued = sum(len(queue) for queue in self.request_queues.values())
            stats['queue_stats'] = {
                'total_queued_requests': total_queued,
                'queued_requests_count': self.queued_requests_count,
                'processed_from_queue_count': self.processed_from_queue_count,
                'queue_timeouts_count': self.queue_timeouts_count,
                'active_queue_processors': sum(
                    1 for task in self.queue_processing_tasks.values() 
                    if not task.done()
                ),
                'max_queue_size': self.config.max_queue_size,
                'queue_timeout': self.config.queue_timeout
            }
        
        return stats
    
    def reset_client(self, client_id: str) -> None:
        """Reset rate limit state for a specific client."""
        self.tokens[client_id] = self.config.requests_per_minute
        self.last_update[client_id] = time.time()
        
        if self.config.enable_burst:
            self.burst_tokens[client_id] = self.config.burst_allowance
        
        if self.config.enable_history_tracking and client_id in self.request_history:
            self.request_history[client_id].clear()
        
        if self.config.enable_request_queuing:
            # Cancel all queued requests
            if client_id in self.request_queues:
                for queued_request in self.request_queues[client_id]:
                    if queued_request.timeout_handle:
                        queued_request.timeout_handle.cancel()
                    if not queued_request.future.done():
                        queued_request.future.set_exception(
                            Exception("Client rate limit reset")
                        )
                self.request_queues[client_id].clear()
            
            # Cancel queue processing task
            if client_id in self.queue_processing_tasks:
                task = self.queue_processing_tasks[client_id]
                if not task.done():
                    task.cancel()
                del self.queue_processing_tasks[client_id]
        
        self.logger.info(f"Rate limit reset for client: {client_id}")
    
    def reset_all_clients(self) -> None:
        """Reset rate limit state for all clients."""
        current_time = time.time()
        
        for client_id in list(self.tokens.keys()):
            self.tokens[client_id] = self.config.requests_per_minute
            self.last_update[client_id] = current_time
            
            if self.config.enable_burst:
                self.burst_tokens[client_id] = self.config.burst_allowance
        
        if self.config.enable_history_tracking:
            for history in self.request_history.values():
                history.clear()
        
        if self.config.enable_request_queuing:
            # Cancel all queued requests
            for client_id, queue in self.request_queues.items():
                for queued_request in queue:
                    if queued_request.timeout_handle:
                        queued_request.timeout_handle.cancel()
                    if not queued_request.future.done():
                        queued_request.future.set_exception(
                            Exception("All clients rate limit reset")
                        )
            self.request_queues.clear()
            
            # Cancel all queue processing tasks
            for task in self.queue_processing_tasks.values():
                if not task.done():
                    task.cancel()
            self.queue_processing_tasks.clear()
            
            # Reset queue statistics
            self.queued_requests_count = 0
            self.processed_from_queue_count = 0
            self.queue_timeouts_count = 0
        
        self.total_requests = 0
        self.blocked_requests = 0
        self.active_clients.clear()
        
        self.logger.info("Rate limit reset for all clients")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old client data."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_old_clients()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_old_clients(self) -> None:
        """Remove data for clients that haven't been active recently."""
        current_time = time.time()
        cleanup_threshold = current_time - (self.config.cleanup_interval * 2)
        
        clients_to_remove = [
            client_id for client_id, last_update in self.last_update.items()
            if last_update < cleanup_threshold
        ]
        
        for client_id in clients_to_remove:
            # Remove from all tracking dictionaries
            self.tokens.pop(client_id, None)
            self.last_update.pop(client_id, None)
            
            if self.config.enable_burst:
                self.burst_tokens.pop(client_id, None)
            
            if self.config.enable_history_tracking:
                self.request_history.pop(client_id, None)
            
            self.active_clients.discard(client_id)
        
        if clients_to_remove:
            self.logger.debug(f"Cleaned up {len(clients_to_remove)} inactive clients")
    
    async def close(self) -> None:
        """Clean shutdown of the rate limiter."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.config.enable_request_queuing:
            # Cancel all queue processing tasks
            for client_id, task in self.queue_processing_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Cancel all queued requests
            for client_id, queue in self.request_queues.items():
                for queued_request in queue:
                    if queued_request.timeout_handle:
                        queued_request.timeout_handle.cancel()
                    if not queued_request.future.done():
                        queued_request.future.set_exception(
                            Exception("Rate limiter shutting down")
                        )
        
        self.logger.info("Rate limiter shutdown complete")


class MultiTierRateLimiter:
    """
    Multi-tier rate limiter supporting different limits for different client types.
    
    Useful for implementing different rate limits for:
    - Free vs premium users
    - Different API endpoints
    - Internal vs external clients
    """
    
    def __init__(self, tier_configs: Dict[str, RateLimitConfig]):
        """
        Initialize multi-tier rate limiter.
        
        Args:
            tier_configs: Dictionary mapping tier names to rate limit configurations
        """
        self.limiters = {
            tier: RateLimiter(config) 
            for tier, config in tier_configs.items()
        }
        self.default_tier = list(tier_configs.keys())[0] if tier_configs else None
        self.logger = logging.getLogger(__name__)
    
    async def is_allowed(
        self, 
        client_id: str, 
        tier: Optional[str] = None,
        request_weight: int = 1
    ) -> bool:
        """
        Check if request is allowed for client in specified tier.
        
        Args:
            client_id: Unique identifier for the client
            tier: Rate limit tier to use (uses default if None)
            request_weight: Weight of the request
            
        Returns:
            True if request is allowed, False if rate limited
        """
        tier = tier or self.default_tier
        
        if tier not in self.limiters:
            self.logger.warning(f"Unknown tier '{tier}', using default '{self.default_tier}'")
            tier = self.default_tier
        
        if tier is None:
            self.logger.error("No rate limit tiers configured")
            return True  # Allow by default if no configuration
        
        return await self.limiters[tier].is_allowed(client_id, request_weight)
    
    async def check_rate_limit(
        self, 
        client_id: str, 
        tier: Optional[str] = None,
        request_weight: int = 1
    ) -> RateLimitInfo:
        """Check rate limit status for client in specified tier."""
        tier = tier or self.default_tier
        
        if tier not in self.limiters:
            tier = self.default_tier
        
        return await self.limiters[tier].check_rate_limit(client_id, request_weight)
    
    def get_tier_stats(self, tier: str) -> Dict[str, any]:
        """Get statistics for a specific tier."""
        if tier in self.limiters:
            return self.limiters[tier].get_global_stats()
        return {}
    
    def get_all_stats(self) -> Dict[str, Dict[str, any]]:
        """Get statistics for all tiers."""
        return {
            tier: limiter.get_global_stats()
            for tier, limiter in self.limiters.items()
        }
    
    async def close(self) -> None:
        """Close all rate limiters."""
        for limiter in self.limiters.values():
            await limiter.close()


# Utility functions for common rate limiting patterns

def create_rate_limiter_from_env() -> RateLimiter:
    """Create rate limiter from environment variables."""
    import os
    
    config = RateLimitConfig(
        requests_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', '100')),
        burst_allowance=int(os.getenv('RATE_LIMIT_BURST', '10')),
        window_seconds=int(os.getenv('RATE_LIMIT_WINDOW', '60')),
        cleanup_interval=int(os.getenv('RATE_LIMIT_CLEANUP_INTERVAL', '300')),
        enable_burst=os.getenv('RATE_LIMIT_ENABLE_BURST', 'true').lower() == 'true',
        enable_history_tracking=os.getenv('RATE_LIMIT_ENABLE_HISTORY', 'true').lower() == 'true',
        enable_request_queuing=os.getenv('RATE_LIMIT_ENABLE_QUEUING', 'false').lower() == 'true',
        max_queue_size=int(os.getenv('RATE_LIMIT_MAX_QUEUE_SIZE', '100')),
        queue_timeout=float(os.getenv('RATE_LIMIT_QUEUE_TIMEOUT', '30.0'))
    )
    
    return RateLimiter(config)


def create_multi_tier_rate_limiter_from_env() -> MultiTierRateLimiter:
    """Create multi-tier rate limiter from environment variables."""
    import os
    
    # Define tier configurations from environment
    tier_configs = {}
    
    # Free tier configuration
    if os.getenv('RATE_LIMIT_FREE_TIER_PER_MINUTE'):
        tier_configs['free'] = RateLimitConfig(
            requests_per_minute=int(os.getenv('RATE_LIMIT_FREE_TIER_PER_MINUTE', '60')),
            burst_allowance=int(os.getenv('RATE_LIMIT_FREE_TIER_BURST', '5')),
            window_seconds=int(os.getenv('RATE_LIMIT_FREE_TIER_WINDOW', '60')),
            cleanup_interval=int(os.getenv('RATE_LIMIT_CLEANUP_INTERVAL', '300')),
            enable_burst=os.getenv('RATE_LIMIT_ENABLE_BURST', 'true').lower() == 'true',
            enable_history_tracking=os.getenv('RATE_LIMIT_ENABLE_HISTORY', 'true').lower() == 'true',
            enable_request_queuing=os.getenv('RATE_LIMIT_ENABLE_QUEUING', 'false').lower() == 'true',
            max_queue_size=int(os.getenv('RATE_LIMIT_MAX_QUEUE_SIZE', '100')),
            queue_timeout=float(os.getenv('RATE_LIMIT_QUEUE_TIMEOUT', '30.0'))
        )
    
    # Premium tier configuration
    if os.getenv('RATE_LIMIT_PREMIUM_TIER_PER_MINUTE'):
        tier_configs['premium'] = RateLimitConfig(
            requests_per_minute=int(os.getenv('RATE_LIMIT_PREMIUM_TIER_PER_MINUTE', '300')),
            burst_allowance=int(os.getenv('RATE_LIMIT_PREMIUM_TIER_BURST', '30')),
            window_seconds=int(os.getenv('RATE_LIMIT_PREMIUM_TIER_WINDOW', '60')),
            cleanup_interval=int(os.getenv('RATE_LIMIT_CLEANUP_INTERVAL', '300')),
            enable_burst=os.getenv('RATE_LIMIT_ENABLE_BURST', 'true').lower() == 'true',
            enable_history_tracking=os.getenv('RATE_LIMIT_ENABLE_HISTORY', 'true').lower() == 'true',
            enable_request_queuing=os.getenv('RATE_LIMIT_ENABLE_QUEUING', 'false').lower() == 'true',
            max_queue_size=int(os.getenv('RATE_LIMIT_MAX_QUEUE_SIZE', '100')),
            queue_timeout=float(os.getenv('RATE_LIMIT_QUEUE_TIMEOUT', '30.0'))
        )
    
    # Enterprise tier configuration
    if os.getenv('RATE_LIMIT_ENTERPRISE_TIER_PER_MINUTE'):
        tier_configs['enterprise'] = RateLimitConfig(
            requests_per_minute=int(os.getenv('RATE_LIMIT_ENTERPRISE_TIER_PER_MINUTE', '1000')),
            burst_allowance=int(os.getenv('RATE_LIMIT_ENTERPRISE_TIER_BURST', '100')),
            window_seconds=int(os.getenv('RATE_LIMIT_ENTERPRISE_TIER_WINDOW', '60')),
            cleanup_interval=int(os.getenv('RATE_LIMIT_CLEANUP_INTERVAL', '300')),
            enable_burst=os.getenv('RATE_LIMIT_ENABLE_BURST', 'true').lower() == 'true',
            enable_history_tracking=os.getenv('RATE_LIMIT_ENABLE_HISTORY', 'true').lower() == 'true',
            enable_request_queuing=os.getenv('RATE_LIMIT_ENABLE_QUEUING', 'false').lower() == 'true',
            max_queue_size=int(os.getenv('RATE_LIMIT_MAX_QUEUE_SIZE', '100')),
            queue_timeout=float(os.getenv('RATE_LIMIT_QUEUE_TIMEOUT', '30.0'))
        )
    
    # Internal tier configuration
    if os.getenv('RATE_LIMIT_INTERNAL_TIER_PER_MINUTE'):
        tier_configs['internal'] = RateLimitConfig(
            requests_per_minute=int(os.getenv('RATE_LIMIT_INTERNAL_TIER_PER_MINUTE', '5000')),
            burst_allowance=int(os.getenv('RATE_LIMIT_INTERNAL_TIER_BURST', '500')),
            window_seconds=int(os.getenv('RATE_LIMIT_INTERNAL_TIER_WINDOW', '60')),
            cleanup_interval=int(os.getenv('RATE_LIMIT_CLEANUP_INTERVAL', '300')),
            enable_burst=os.getenv('RATE_LIMIT_ENABLE_BURST', 'true').lower() == 'true',
            enable_history_tracking=os.getenv('RATE_LIMIT_ENABLE_HISTORY', 'true').lower() == 'true',
            enable_request_queuing=os.getenv('RATE_LIMIT_ENABLE_QUEUING', 'false').lower() == 'true',
            max_queue_size=int(os.getenv('RATE_LIMIT_MAX_QUEUE_SIZE', '100')),
            queue_timeout=float(os.getenv('RATE_LIMIT_QUEUE_TIMEOUT', '30.0'))
        )
    
    # If no tiers configured, create default tier
    if not tier_configs:
        tier_configs['default'] = RateLimitConfig(
            requests_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', '100')),
            burst_allowance=int(os.getenv('RATE_LIMIT_BURST', '10')),
            window_seconds=int(os.getenv('RATE_LIMIT_WINDOW', '60')),
            cleanup_interval=int(os.getenv('RATE_LIMIT_CLEANUP_INTERVAL', '300')),
            enable_burst=os.getenv('RATE_LIMIT_ENABLE_BURST', 'true').lower() == 'true',
            enable_history_tracking=os.getenv('RATE_LIMIT_ENABLE_HISTORY', 'true').lower() == 'true',
            enable_request_queuing=os.getenv('RATE_LIMIT_ENABLE_QUEUING', 'false').lower() == 'true',
            max_queue_size=int(os.getenv('RATE_LIMIT_MAX_QUEUE_SIZE', '100')),
            queue_timeout=float(os.getenv('RATE_LIMIT_QUEUE_TIMEOUT', '30.0'))
        )
    
    return MultiTierRateLimiter(tier_configs)


def create_client_type_rate_limiter_from_env() -> MultiTierRateLimiter:
    """Create rate limiter with different limits per client type from environment variables."""
    import os
    
    # Define client type configurations from environment
    client_type_configs = {}
    
    # IP-based client configuration
    client_type_configs['ip_based'] = RateLimitConfig(
        requests_per_minute=int(os.getenv('RATE_LIMIT_IP_BASED_PER_MINUTE', '100')),
        burst_allowance=int(os.getenv('RATE_LIMIT_IP_BASED_BURST', '10')),
        window_seconds=int(os.getenv('RATE_LIMIT_WINDOW', '60')),
        cleanup_interval=int(os.getenv('RATE_LIMIT_CLEANUP_INTERVAL', '300')),
        enable_burst=os.getenv('RATE_LIMIT_ENABLE_BURST', 'true').lower() == 'true',
        enable_history_tracking=os.getenv('RATE_LIMIT_ENABLE_HISTORY', 'true').lower() == 'true',
        enable_request_queuing=os.getenv('RATE_LIMIT_ENABLE_QUEUING', 'false').lower() == 'true',
        max_queue_size=int(os.getenv('RATE_LIMIT_MAX_QUEUE_SIZE', '100')),
        queue_timeout=float(os.getenv('RATE_LIMIT_QUEUE_TIMEOUT', '30.0'))
    )
    
    # Session-based client configuration
    client_type_configs['session_based'] = RateLimitConfig(
        requests_per_minute=int(os.getenv('RATE_LIMIT_SESSION_BASED_PER_MINUTE', '200')),
        burst_allowance=int(os.getenv('RATE_LIMIT_SESSION_BASED_BURST', '20')),
        window_seconds=int(os.getenv('RATE_LIMIT_WINDOW', '60')),
        cleanup_interval=int(os.getenv('RATE_LIMIT_CLEANUP_INTERVAL', '300')),
        enable_burst=os.getenv('RATE_LIMIT_ENABLE_BURST', 'true').lower() == 'true',
        enable_history_tracking=os.getenv('RATE_LIMIT_ENABLE_HISTORY', 'true').lower() == 'true',
        enable_request_queuing=os.getenv('RATE_LIMIT_ENABLE_QUEUING', 'false').lower() == 'true',
        max_queue_size=int(os.getenv('RATE_LIMIT_MAX_QUEUE_SIZE', '100')),
        queue_timeout=float(os.getenv('RATE_LIMIT_QUEUE_TIMEOUT', '30.0'))
    )
    
    # API key-based client configuration
    client_type_configs['api_key_based'] = RateLimitConfig(
        requests_per_minute=int(os.getenv('RATE_LIMIT_API_KEY_BASED_PER_MINUTE', '500')),
        burst_allowance=int(os.getenv('RATE_LIMIT_API_KEY_BASED_BURST', '50')),
        window_seconds=int(os.getenv('RATE_LIMIT_WINDOW', '60')),
        cleanup_interval=int(os.getenv('RATE_LIMIT_CLEANUP_INTERVAL', '300')),
        enable_burst=os.getenv('RATE_LIMIT_ENABLE_BURST', 'true').lower() == 'true',
        enable_history_tracking=os.getenv('RATE_LIMIT_ENABLE_HISTORY', 'true').lower() == 'true',
        enable_request_queuing=os.getenv('RATE_LIMIT_ENABLE_QUEUING', 'false').lower() == 'true',
        max_queue_size=int(os.getenv('RATE_LIMIT_MAX_QUEUE_SIZE', '100')),
        queue_timeout=float(os.getenv('RATE_LIMIT_QUEUE_TIMEOUT', '30.0'))
    )
    
    return MultiTierRateLimiter(client_type_configs)


@dataclass
class EnhancedRateLimitConfig:
    """Enhanced rate limit configuration with per-client type and tier support."""
    
    # Global settings
    enabled: bool = True
    default_client_type: RateLimitType = RateLimitType.IP_BASED
    default_tier: str = "free"
    strict_mode: bool = False
    fail_open: bool = True
    log_violations: bool = True
    
    # Header configuration
    include_headers: bool = True
    header_prefix: str = "X-RateLimit"
    expose_retry_after: bool = True
    client_id_header: str = "X-Client-ID"
    tier_header: str = "X-Client-Tier"
    
    # Client identification
    trust_forwarded_for: bool = True
    max_forwarded_depth: int = 3
    
    # Security
    whitelist_bypass: bool = True
    blacklist_block: bool = True
    ip_whitelist: List[str] = None
    ip_blacklist: List[str] = None
    
    def __post_init__(self):
        if self.ip_whitelist is None:
            self.ip_whitelist = []
        if self.ip_blacklist is None:
            self.ip_blacklist = []
    
    @classmethod
    def from_env(cls) -> 'EnhancedRateLimitConfig':
        """Load enhanced rate limit configuration from environment variables."""
        import os
        
        # Parse IP lists from environment
        ip_whitelist = []
        whitelist_str = os.getenv('IP_WHITELIST', '').strip()
        if whitelist_str:
            ip_whitelist = [ip.strip() for ip in whitelist_str.split(',') if ip.strip()]
        
        ip_blacklist = []
        blacklist_str = os.getenv('IP_BLACKLIST', '').strip()
        if blacklist_str:
            ip_blacklist = [ip.strip() for ip in blacklist_str.split(',') if ip.strip()]
        
        # Parse default client type
        default_client_type_str = os.getenv('RATE_LIMIT_DEFAULT_CLIENT_TYPE', 'ip_based').lower()
        default_client_type = RateLimitType.IP_BASED
        if default_client_type_str == 'session_based':
            default_client_type = RateLimitType.SESSION_BASED
        elif default_client_type_str == 'api_key_based':
            default_client_type = RateLimitType.API_KEY_BASED
        elif default_client_type_str == 'combined':
            default_client_type = RateLimitType.COMBINED
        
        return cls(
            enabled=os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true',
            default_client_type=default_client_type,
            default_tier=os.getenv('RATE_LIMIT_DEFAULT_TIER', 'free'),
            strict_mode=os.getenv('RATE_LIMIT_STRICT_MODE', 'false').lower() == 'true',
            fail_open=os.getenv('RATE_LIMIT_FAIL_OPEN', 'true').lower() == 'true',
            log_violations=os.getenv('RATE_LIMIT_LOG_VIOLATIONS', 'true').lower() == 'true',
            include_headers=os.getenv('RATE_LIMIT_INCLUDE_HEADERS', 'true').lower() == 'true',
            header_prefix=os.getenv('RATE_LIMIT_HEADER_PREFIX', 'X-RateLimit'),
            expose_retry_after=os.getenv('RATE_LIMIT_EXPOSE_RETRY_AFTER', 'true').lower() == 'true',
            client_id_header=os.getenv('RATE_LIMIT_CLIENT_ID_HEADER', 'X-Client-ID'),
            tier_header=os.getenv('RATE_LIMIT_TIER_HEADER', 'X-Client-Tier'),
            trust_forwarded_for=os.getenv('RATE_LIMIT_TRUST_FORWARDED_FOR', 'true').lower() == 'true',
            max_forwarded_depth=int(os.getenv('RATE_LIMIT_MAX_FORWARDED_DEPTH', '3')),
            whitelist_bypass=os.getenv('RATE_LIMIT_WHITELIST_BYPASS', 'true').lower() == 'true',
            blacklist_block=os.getenv('RATE_LIMIT_BLACKLIST_BLOCK', 'true').lower() == 'true',
            ip_whitelist=ip_whitelist,
            ip_blacklist=ip_blacklist
        )


class EnhancedRateLimiter:
    """
    Enhanced rate limiter with comprehensive per-client configuration support.
    
    Features:
    - Multiple rate limiting strategies (per client type, per tier)
    - IP whitelist/blacklist support
    - Configurable headers and client identification
    - Environment-based configuration
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, config: Optional[EnhancedRateLimitConfig] = None):
        """
        Initialize enhanced rate limiter.
        
        Args:
            config: Enhanced rate limit configuration
        """
        self.config = config or EnhancedRateLimitConfig.from_env()
        
        # Create multi-tier rate limiter for client types
        self.client_type_limiter = create_client_type_rate_limiter_from_env()
        
        # Create multi-tier rate limiter for user tiers
        self.tier_limiter = create_multi_tier_rate_limiter_from_env()
        
        # Fallback single rate limiter
        self.fallback_limiter = create_rate_limiter_from_env()
        
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"Enhanced rate limiter initialized: "
            f"enabled={self.config.enabled}, "
            f"default_client_type={self.config.default_client_type.value}, "
            f"default_tier={self.config.default_tier}"
        )
    
    async def is_allowed(
        self,
        request,
        client_id: Optional[str] = None,
        client_type: Optional[RateLimitType] = None,
        tier: Optional[str] = None,
        request_weight: int = 1
    ) -> bool:
        """
        Check if request is allowed with enhanced client identification and tiering.
        
        Args:
            request: HTTP request object (for header extraction)
            client_id: Override client ID (if None, extracted from request)
            client_type: Override client type (if None, uses default)
            tier: Override tier (if None, extracted from headers or uses default)
            request_weight: Weight of the request
            
        Returns:
            True if request is allowed, False if rate limited
        """
        # Check if rate limiting is enabled
        if not self.config.enabled:
            return True
        
        # Extract client information
        if client_id is None:
            client_type = client_type or self.config.default_client_type
            client_id = self._extract_client_id(request, client_type)
        
        # Check IP whitelist/blacklist
        ip_access_result = self._check_ip_access(request)
        if not ip_access_result:
            return False
        elif ip_access_result == "whitelisted":
            # IP is whitelisted, bypass rate limiting entirely
            return True
        
        # Extract tier from headers or use default
        if tier is None:
            tier = self._extract_tier(request) or self.config.default_tier
        
        try:
            # Try tier-based rate limiting first
            if tier and tier in self.tier_limiter.limiters:
                allowed = await self.tier_limiter.is_allowed(
                    client_id, tier=tier, request_weight=request_weight
                )
                if self.config.log_violations and not allowed:
                    self.logger.warning(
                        f"Rate limit exceeded for client {client_id} "
                        f"(tier: {tier}, type: {client_type.value if client_type else 'unknown'})"
                    )
                return allowed
            
            # Try client type-based rate limiting
            if client_type and client_type.value in self.client_type_limiter.limiters:
                allowed = await self.client_type_limiter.is_allowed(
                    client_id, tier=client_type.value, request_weight=request_weight
                )
                if self.config.log_violations and not allowed:
                    self.logger.warning(
                        f"Rate limit exceeded for client {client_id} "
                        f"(type: {client_type.value})"
                    )
                return allowed
            
            # Fallback to default rate limiter
            allowed = await self.fallback_limiter.is_allowed(
                client_id, request_weight=request_weight
            )
            if self.config.log_violations and not allowed:
                self.logger.warning(
                    f"Rate limit exceeded for client {client_id} (fallback limiter)"
                )
            return allowed
            
        except Exception as e:
            self.logger.error(f"Rate limiting error for client {client_id}: {e}")
            # Fail open or closed based on configuration
            return self.config.fail_open
    
    async def check_rate_limit(
        self,
        request,
        client_id: Optional[str] = None,
        client_type: Optional[RateLimitType] = None,
        tier: Optional[str] = None,
        request_weight: int = 1
    ) -> RateLimitInfo:
        """
        Check rate limit status without consuming tokens.
        
        Args:
            request: HTTP request object
            client_id: Override client ID
            client_type: Override client type
            tier: Override tier
            request_weight: Weight of the request to check
            
        Returns:
            RateLimitInfo with current status
        """
        # Extract client information
        if client_id is None:
            client_type = client_type or self.config.default_client_type
            client_id = self._extract_client_id(request, client_type)
        
        # Extract tier from headers or use default
        if tier is None:
            tier = self._extract_tier(request) or self.config.default_tier
        
        try:
            # Try tier-based rate limiting first
            if tier and tier in self.tier_limiter.limiters:
                return await self.tier_limiter.check_rate_limit(
                    client_id, tier=tier, request_weight=request_weight
                )
            
            # Try client type-based rate limiting
            if client_type and client_type.value in self.client_type_limiter.limiters:
                return await self.client_type_limiter.check_rate_limit(
                    client_id, tier=client_type.value, request_weight=request_weight
                )
            
            # Fallback to default rate limiter
            return await self.fallback_limiter.check_rate_limit(
                client_id, request_weight=request_weight
            )
            
        except Exception as e:
            self.logger.error(f"Rate limit check error for client {client_id}: {e}")
            # Return permissive info on error
            return RateLimitInfo(
                remaining_requests=1000,
                reset_time=0.0,
                limit_per_minute=100,
                window_seconds=60,
                burst_allowance=10
            )
    
    def get_rate_limit_headers(
        self,
        request,
        client_id: Optional[str] = None,
        client_type: Optional[RateLimitType] = None,
        tier: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate rate limit headers for HTTP response.
        
        Args:
            request: HTTP request object
            client_id: Override client ID
            client_type: Override client type
            tier: Override tier
            
        Returns:
            Dictionary of headers to include in response
        """
        if not self.config.include_headers:
            return {}
        
        try:
            # Get rate limit info
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't await here
                # Return basic headers
                return {
                    f"{self.config.header_prefix}-Limit": "100",
                    f"{self.config.header_prefix}-Remaining": "99",
                    f"{self.config.header_prefix}-Reset": str(int(time.time() + 60))
                }
            else:
                # We can run async code
                info = loop.run_until_complete(
                    self.check_rate_limit(request, client_id, client_type, tier)
                )
        except Exception as e:
            self.logger.error(f"Error generating rate limit headers: {e}")
            return {}
        
        headers = {
            f"{self.config.header_prefix}-Limit": str(info.limit_per_minute),
            f"{self.config.header_prefix}-Remaining": str(info.remaining_requests),
            f"{self.config.header_prefix}-Reset": str(int(time.time() + info.reset_time))
        }
        
        if self.config.expose_retry_after and info.remaining_requests <= 0:
            headers["Retry-After"] = str(int(info.reset_time))
        
        return headers
    
    def _extract_client_id(self, request, client_type: RateLimitType) -> str:
        """Extract client ID from request based on client type."""
        # Check for explicit client ID header first
        if hasattr(request, 'headers') and request.headers:
            explicit_id = request.headers.get(self.config.client_id_header)
            if explicit_id:
                return f"explicit:{explicit_id}"
        
        # Use the existing utility function
        return get_client_id_from_request(request, client_type)
    
    def _extract_tier(self, request) -> Optional[str]:
        """Extract tier from request headers."""
        if hasattr(request, 'headers') and request.headers:
            return request.headers.get(self.config.tier_header)
        return None
    
    def _check_ip_access(self, request):
        """Check if client IP is allowed based on whitelist/blacklist."""
        if not self.config.ip_whitelist and not self.config.ip_blacklist:
            return True
        
        # Extract client IP
        client_ip = self._extract_client_ip(request)
        if not client_ip:
            return True  # Allow if we can't determine IP
        
        self.logger.debug(f"Checking IP access for {client_ip}, whitelist: {self.config.ip_whitelist}, blacklist: {self.config.ip_blacklist}")
        
        # Check blacklist first
        if self.config.ip_blacklist and self.config.blacklist_block:
            if client_ip in self.config.ip_blacklist:
                self.logger.warning(f"Blocked request from blacklisted IP: {client_ip}")
                return False
        
        # Check whitelist
        if self.config.ip_whitelist:
            if self.config.whitelist_bypass and client_ip in self.config.ip_whitelist:
                self.logger.debug(f"Allowing whitelisted IP: {client_ip}")
                return "whitelisted"  # Special return value to bypass rate limiting
            elif client_ip not in self.config.ip_whitelist:
                # If whitelist exists but IP not in it, block
                self.logger.warning(f"Blocked request from non-whitelisted IP: {client_ip}")
                return False
        
        return True
    
    def _extract_client_ip(self, request) -> Optional[str]:
        """Extract client IP from request."""
        # Check X-Forwarded-For if trusted
        if self.config.trust_forwarded_for and hasattr(request, 'headers'):
            headers = getattr(request, 'headers', {})
            forwarded_for = headers.get('X-Forwarded-For') if headers else None
            if forwarded_for:
                # Take the first IP in the chain (original client)
                ips = [ip.strip() for ip in forwarded_for.split(',')]
                if ips and len(ips) <= self.config.max_forwarded_depth:
                    self.logger.debug(f"Using forwarded IP: {ips[0]}")
                    return ips[0]
        
        # Fallback to direct client IP
        if hasattr(request, 'client'):
            client = getattr(request, 'client')
            if isinstance(client, dict):
                # Handle dict-style client (like in tests)
                client_ip = client.get('host')
            else:
                # Handle object-style client
                client_ip = getattr(client, 'host', None)
            
            self.logger.debug(f"Using direct client IP: {client_ip}")
            return client_ip
        
        self.logger.debug("No client IP found")
        return None
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all rate limiters."""
        stats = {
            'config': {
                'enabled': self.config.enabled,
                'default_client_type': self.config.default_client_type.value,
                'default_tier': self.config.default_tier,
                'strict_mode': self.config.strict_mode,
                'fail_open': self.config.fail_open
            },
            'client_type_stats': self.client_type_limiter.get_all_stats(),
            'tier_stats': self.tier_limiter.get_all_stats(),
            'fallback_stats': self.fallback_limiter.get_global_stats()
        }
        
        return stats
    
    async def close(self) -> None:
        """Close all rate limiters."""
        await self.client_type_limiter.close()
        await self.tier_limiter.close()
        await self.fallback_limiter.close()


def get_client_id_from_request(request, rate_limit_type: RateLimitType = RateLimitType.IP_BASED) -> str:
    """
    Extract client ID from request based on rate limit type.
    
    Args:
        request: HTTP request object
        rate_limit_type: Type of client identification to use
        
    Returns:
        Client identifier string
    """
    if rate_limit_type == RateLimitType.IP_BASED:
        # Get client IP, handling proxies
        headers = getattr(request, 'headers', None)
        forwarded_for = headers.get('X-Forwarded-For') if headers else None
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        client = getattr(request, 'client', None)
        if isinstance(client, dict):
            return client.get('host', 'unknown')
        elif client:
            return getattr(client, 'host', 'unknown')
        return 'unknown'
    
    elif rate_limit_type == RateLimitType.SESSION_BASED:
        # Use session ID if available
        session = getattr(request, 'session', None)
        session_id = session.get('id') if isinstance(session, dict) else getattr(session, 'id', None) if session else None
        if session_id:
            return f"session:{session_id}"
        # Fallback to IP
        return get_client_id_from_request(request, RateLimitType.IP_BASED)
    
    elif rate_limit_type == RateLimitType.API_KEY_BASED:
        # Use API key from headers
        headers = getattr(request, 'headers', None)
        if headers:
            api_key = headers.get('X-API-Key') or headers.get('Authorization', '').replace('Bearer ', '')
            if api_key:
                return f"api_key:{api_key[:8]}..."  # Truncate for privacy
        # Fallback to IP
        return get_client_id_from_request(request, RateLimitType.IP_BASED)
    
    elif rate_limit_type == RateLimitType.COMBINED:
        # Combine multiple identifiers
        ip = get_client_id_from_request(request, RateLimitType.IP_BASED)
        session = getattr(request, 'session', None)
        session_id = session.get('id') if isinstance(session, dict) else getattr(session, 'id', None) if session else 'no_session'
        return f"{ip}:{session_id}"
    
    else:
        # Default to IP-based
        return get_client_id_from_request(request, RateLimitType.IP_BASED)


# Global enhanced rate limiter instance
enhanced_rate_limiter: Optional[EnhancedRateLimiter] = None


def get_enhanced_rate_limiter() -> EnhancedRateLimiter:
    """Get or create the global enhanced rate limiter instance."""
    global enhanced_rate_limiter
    
    if enhanced_rate_limiter is None:
        enhanced_rate_limiter = EnhancedRateLimiter()
    
    return enhanced_rate_limiter
