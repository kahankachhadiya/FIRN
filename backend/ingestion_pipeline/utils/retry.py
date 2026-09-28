"""
Retry utilities for database operations with exponential backoff.

This module provides decorators and utilities for adding retry logic to
database operations with configurable retry attempts, delays, and error handling.
"""

import time
import functools
import logging
from typing import Callable, Type, Tuple, Optional, Any
from utils.errors import DatabaseError, TimeoutError

# Import Elasticsearch exception types
try:
    from elasticsearch.exceptions import ConnectionError as ElasticsearchConnectionError
    from elasticsearch.exceptions import TransportError, ConnectionTimeout
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    # Elasticsearch not installed, define placeholder exceptions
    ElasticsearchConnectionError = type('ElasticsearchConnectionError', (Exception,), {})
    TransportError = type('TransportError', (Exception,), {})
    ConnectionTimeout = type('ConnectionTimeout', (Exception,), {})
    ELASTICSEARCH_AVAILABLE = False


logger = logging.getLogger(__name__)


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0
) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
    
    Returns:
        Delay in seconds, capped at max_delay
    """
    delay = base_delay * (exponential_base ** attempt)
    return min(delay, max_delay)


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (DatabaseError,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator to add retry logic with exponential backoff to a function.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        retryable_exceptions: Tuple of exception types that should trigger a retry
        on_retry: Optional callback function called on each retry with (exception, attempt)
    
    Returns:
        Decorated function with retry logic
    
    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        def query_database():
            # Database operation that may fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            func_name = getattr(func, '__name__', 'unknown_function')
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # Don't retry on last attempt
                    if attempt >= max_retries:
                        logger.error(
                            f"{func_name} failed after {max_retries} retries: {e}"
                        )
                        raise
                    
                    # Calculate backoff delay
                    delay = exponential_backoff(attempt, base_delay, max_delay)
                    
                    logger.warning(
                        f"{func_name} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)
                    
                    # Wait before retry
                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception, raise immediately
                    logger.error(f"{func_name} failed with non-retryable error: {e}")
                    raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to a function.
    
    Note: This is a simple timeout implementation. For async operations,
    consider using asyncio.wait_for() instead.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
    
    Returns:
        Decorated function with timeout
    
    Example:
        @with_timeout(30.0)
        def slow_operation():
            # Operation that might take too long
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"{func.__name__} exceeded timeout of {timeout_seconds}s"
                )
            
            # Set timeout alarm (Unix only)
            try:
                # Try to use signal-based timeout (Unix)
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                
                return result
            except (AttributeError, ValueError):
                # signal.SIGALRM not available (Windows), just execute without timeout
                logger.debug(f"Timeout not available on this platform, executing {func.__name__} without timeout")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retryable_exceptions: Tuple[Type[Exception], ...] = (DatabaseError,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions


# Default retry configurations for different database types

QDRANT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=15.0,
    retryable_exceptions=(DatabaseError, ConnectionError, TimeoutError)
)

FALKOR_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(DatabaseError, ConnectionError, TimeoutError)
)

ELASTICSEARCH_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(
        DatabaseError,
        ElasticsearchConnectionError,
        TransportError,
        ConnectionTimeout,
        TimeoutError
    )
)
