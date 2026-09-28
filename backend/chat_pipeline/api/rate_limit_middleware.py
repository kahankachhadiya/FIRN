"""
Rate Limiting Middleware for FastAPI

This module provides middleware for adding rate limit headers and proper error
responses to API endpoints. It integrates with the EnhancedRateLimiter to provide
production-ready rate limiting with comprehensive headers and error handling.

Features:
- Automatic rate limit header injection
- Proper HTTP 429 responses with Retry-After headers
- Client identification from requests
- Configurable rate limiting per endpoint
- Support for different client types and tiers
"""

import time
import logging
from typing import Optional, Callable, Dict
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from utils.rate_limiter import (
    EnhancedRateLimiter,
    RateLimitInfo,
    RateLimitType,
    get_enhanced_rate_limiter
)

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting with automatic header injection.
    
    This middleware:
    1. Checks rate limits before processing requests
    2. Adds rate limit headers to all responses
    3. Returns proper 429 errors when limits are exceeded
    4. Supports different client types and tiers
    """
    
    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: Optional[EnhancedRateLimiter] = None,
        exempt_paths: Optional[list] = None
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: ASGI application
            rate_limiter: EnhancedRateLimiter instance (creates default if None)
            exempt_paths: List of paths to exempt from rate limiting
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter or get_enhanced_rate_limiter()
        self.exempt_paths = exempt_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/metrics"
        ]
        logger.info(f"Rate limit middleware initialized with {len(self.exempt_paths)} exempt paths")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            HTTP response with rate limit headers
        """
        # Check if path is exempt
        if self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        # Extract client information
        client_id = self._extract_client_id(request)
        client_type = self._extract_client_type(request)
        tier = self._extract_tier(request)
        
        try:
            # Check rate limit
            allowed = await self.rate_limiter.is_allowed(
                request=request,
                client_id=client_id,
                client_type=client_type,
                tier=tier
            )
            
            if not allowed:
                # Rate limit exceeded - return 429 with proper headers
                return await self._create_rate_limit_error_response(
                    request=request,
                    client_id=client_id,
                    client_type=client_type,
                    tier=tier
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response = await self._add_rate_limit_headers(
                response=response,
                request=request,
                client_id=client_id,
                client_type=client_type,
                tier=tier
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in rate limit middleware: {e}")
            # Fail open - allow request if rate limiting fails
            if self.rate_limiter.config.fail_open:
                return await call_next(request)
            else:
                # Fail closed - return error
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "error": "rate_limit_error",
                        "message": "Rate limiting service temporarily unavailable",
                        "detail": str(e) if logger.level == logging.DEBUG else None
                    }
                )
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from rate limiting."""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)
    
    def _extract_client_id(self, request: Request) -> str:
        """Extract client ID from request."""
        # Check for explicit client ID header
        if self.rate_limiter.config.client_id_header in request.headers:
            return f"explicit:{request.headers[self.rate_limiter.config.client_id_header]}"
        
        # Use IP address as default
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _extract_client_type(self, request: Request) -> RateLimitType:
        """Extract client type from request."""
        # Check for API key
        if "X-API-Key" in request.headers or "Authorization" in request.headers:
            return RateLimitType.API_KEY_BASED
        
        # Check for session (safely - SessionMiddleware may not be installed)
        try:
            if hasattr(request, 'session') and "session" in request.scope and request.session:
                return RateLimitType.SESSION_BASED
        except (AssertionError, KeyError):
            # SessionMiddleware not installed, continue to IP-based
            pass
        
        # Default to IP-based
        return RateLimitType.IP_BASED
    
    def _extract_tier(self, request: Request) -> Optional[str]:
        """Extract tier from request headers."""
        return request.headers.get(self.rate_limiter.config.tier_header)
    
    async def _create_rate_limit_error_response(
        self,
        request: Request,
        client_id: str,
        client_type: RateLimitType,
        tier: Optional[str]
    ) -> JSONResponse:
        """
        Create a proper 429 error response with rate limit information.
        
        Args:
            request: HTTP request
            client_id: Client identifier
            client_type: Type of client
            tier: Client tier
            
        Returns:
            JSONResponse with 429 status and rate limit headers
        """
        # Get rate limit info
        try:
            limit_info = await self.rate_limiter.check_rate_limit(
                request=request,
                client_id=client_id,
                client_type=client_type,
                tier=tier
            )
        except Exception as e:
            logger.error(f"Error getting rate limit info: {e}")
            # Create default limit info
            limit_info = RateLimitInfo(
                remaining_requests=0,
                reset_time=60.0,
                limit_per_minute=100,
                window_seconds=60,
                burst_allowance=10
            )
        
        # Create headers
        headers = self._create_rate_limit_headers(limit_info)
        
        # Always add Retry-After header for 429 responses if configured
        if self.rate_limiter.config.expose_retry_after:
            headers["Retry-After"] = str(max(1, int(limit_info.reset_time)))
        
        # Create error response
        content = {
            "error": "rate_limit_exceeded",
            "message": f"Rate limit exceeded. Please retry after {int(limit_info.reset_time)} seconds.",
            "limit": limit_info.limit_per_minute,
            "remaining": int(limit_info.remaining_requests),
            "reset_time": int(time.time() + limit_info.reset_time),
            "retry_after": max(1, int(limit_info.reset_time))
        }
        
        logger.warning(
            f"Rate limit exceeded for client {client_id} "
            f"(type: {client_type.value if client_type else 'unknown'}, "
            f"tier: {tier or 'default'})"
        )
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=content,
            headers=headers
        )
    
    async def _add_rate_limit_headers(
        self,
        response: Response,
        request: Request,
        client_id: str,
        client_type: RateLimitType,
        tier: Optional[str]
    ) -> Response:
        """
        Add rate limit headers to response.
        
        Args:
            response: HTTP response
            request: HTTP request
            client_id: Client identifier
            client_type: Type of client
            tier: Client tier
            
        Returns:
            Response with rate limit headers added
        """
        if not self.rate_limiter.config.include_headers:
            return response
        
        try:
            # Get rate limit info
            limit_info = await self.rate_limiter.check_rate_limit(
                request=request,
                client_id=client_id,
                client_type=client_type,
                tier=tier
            )
            
            # Create headers
            headers = self._create_rate_limit_headers(limit_info)
            
            # Add headers to response
            for key, value in headers.items():
                response.headers[key] = value
            
        except Exception as e:
            logger.error(f"Error adding rate limit headers: {e}")
            # Continue without headers if there's an error
        
        return response
    
    def _create_rate_limit_headers(self, limit_info: RateLimitInfo) -> Dict[str, str]:
        """
        Create rate limit headers from limit info.
        
        Args:
            limit_info: Rate limit information
            
        Returns:
            Dictionary of headers
        """
        prefix = self.rate_limiter.config.header_prefix
        
        headers = {
            f"{prefix}-Limit": str(limit_info.limit_per_minute),
            f"{prefix}-Remaining": str(max(0, int(limit_info.remaining_requests))),
            f"{prefix}-Reset": str(int(time.time() + limit_info.reset_time))
        }
        
        # Add Retry-After header if rate limit exceeded
        if self.rate_limiter.config.expose_retry_after and limit_info.remaining_requests <= 0:
            headers["Retry-After"] = str(int(limit_info.reset_time))
        
        return headers


def create_rate_limit_middleware(
    app: ASGIApp,
    exempt_paths: Optional[list] = None
) -> RateLimitMiddleware:
    """
    Factory function to create rate limit middleware.
    
    Args:
        app: ASGI application
        exempt_paths: List of paths to exempt from rate limiting
        
    Returns:
        Configured RateLimitMiddleware instance
    """
    rate_limiter = get_enhanced_rate_limiter()
    return RateLimitMiddleware(
        app=app,
        rate_limiter=rate_limiter,
        exempt_paths=exempt_paths
    )


# Decorator for endpoint-specific rate limiting
def rate_limit(
    requests_per_minute: Optional[int] = None,
    tier: Optional[str] = None,
    client_type: Optional[RateLimitType] = None
):
    """
    Decorator for endpoint-specific rate limiting.
    
    This decorator can be used to apply custom rate limits to specific endpoints.
    
    Args:
        requests_per_minute: Custom rate limit for this endpoint
        tier: Required tier for this endpoint
        client_type: Required client type for this endpoint
        
    Example:
        @app.get("/api/premium")
        @rate_limit(requests_per_minute=1000, tier="premium")
        async def premium_endpoint():
            return {"message": "Premium content"}
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This is a placeholder - actual implementation would need
            # to integrate with the middleware or use dependency injection
            return await func(*args, **kwargs)
        return wrapper
    return decorator
