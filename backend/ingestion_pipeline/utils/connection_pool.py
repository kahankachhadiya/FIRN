"""
Connection pooling and retry logic for graph databases and other services.

This module provides connection pooling, retry logic, timeout handling,
and graceful degradation for all database connections and external services.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from functools import wraps

from utils.logging_utils import get_logger
from utils.errors import DatabaseError, TimeoutError

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class ConnectionConfig:
    """Configuration for connection pooling and retry logic."""
    max_connections: int = 10
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay for exponential backoff
    max_delay: float = 30.0  # Maximum delay between retries
    timeout: float = 30.0    # Default timeout for operations
    health_check_interval: float = 60.0  # Health check interval in seconds
    connection_timeout: float = 10.0  # Timeout for establishing connections


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add random jitter to prevent thundering herd


class ConnectionPool:
    """Generic connection pool with health checking and retry logic."""
    
    def __init__(
        self,
        name: str,
        create_connection: Callable[[], Any],
        validate_connection: Callable[[Any], bool],
        close_connection: Callable[[Any], None],
        config: ConnectionConfig
    ):
        """
        Initialize connection pool.
        
        Args:
            name: Name of the connection pool for logging
            create_connection: Function to create new connections
            validate_connection: Function to validate connection health
            close_connection: Function to close connections
            config: Connection pool configuration
        """
        self.name = name
        self.create_connection = create_connection
        self.validate_connection = validate_connection
        self.close_connection = close_connection
        self.config = config
        
        self._pool: List[Any] = []
        self._pool_lock = threading.Lock()
        self._active_connections = 0
        self._last_health_check = 0.0
        self._health_check_lock = threading.Lock()
        self._is_healthy = True
        
        logger.info(f"Initialized connection pool '{name}' with max_connections={config.max_connections}")
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool with automatic cleanup.
        
        Yields:
            Connection object
            
        Raises:
            DatabaseError: If unable to get a valid connection
        """
        connection = None
        try:
            connection = self._acquire_connection()
            yield connection
        finally:
            if connection:
                self._release_connection(connection)
    
    def _acquire_connection(self) -> Any:
        """Acquire a connection from the pool or create a new one."""
        with self._pool_lock:
            # Try to get a connection from the pool
            while self._pool:
                connection = self._pool.pop()
                if self.validate_connection(connection):
                    self._active_connections += 1
                    logger.debug(f"Acquired connection from pool '{self.name}' (active: {self._active_connections})")
                    return connection
                else:
                    # Connection is invalid, close it
                    try:
                        self.close_connection(connection)
                    except Exception as e:
                        logger.debug(f"Error closing invalid connection: {e}")
            
            # No valid connections in pool, create a new one
            if self._active_connections < self.config.max_connections:
                try:
                    connection = self.create_connection()
                    if self.validate_connection(connection):
                        self._active_connections += 1
                        logger.debug(f"Created new connection for pool '{self.name}' (active: {self._active_connections})")
                        return connection
                    else:
                        try:
                            self.close_connection(connection)
                        except Exception:
                            pass
                        raise DatabaseError(f"Failed to create valid connection for pool '{self.name}'")
                except Exception as e:
                    raise DatabaseError(f"Failed to create connection for pool '{self.name}': {e}") from e
            else:
                raise DatabaseError(f"Connection pool '{self.name}' exhausted (max: {self.config.max_connections})")
    
    def _release_connection(self, connection: Any) -> None:
        """Release a connection back to the pool."""
        with self._pool_lock:
            self._active_connections -= 1
            
            if self.validate_connection(connection) and len(self._pool) < self.config.max_connections:
                self._pool.append(connection)
                logger.debug(f"Released connection to pool '{self.name}' (pooled: {len(self._pool)})")
            else:
                # Connection is invalid or pool is full, close it
                try:
                    self.close_connection(connection)
                    logger.debug(f"Closed connection for pool '{self.name}' (invalid or pool full)")
                except Exception as e:
                    logger.debug(f"Error closing connection: {e}")
    
    def health_check(self) -> bool:
        """
        Perform health check on the connection pool.
        
        Returns:
            True if pool is healthy, False otherwise
        """
        current_time = time.time()
        
        with self._health_check_lock:
            # Skip if health check was done recently
            if current_time - self._last_health_check < self.config.health_check_interval:
                return self._is_healthy
            
            self._last_health_check = current_time
        
        try:
            # Try to create and validate a test connection
            test_connection = self.create_connection()
            is_healthy = self.validate_connection(test_connection)
            
            try:
                self.close_connection(test_connection)
            except Exception:
                pass
            
            self._is_healthy = is_healthy
            
            if is_healthy:
                logger.debug(f"Health check passed for pool '{self.name}'")
            else:
                logger.warning(f"Health check failed for pool '{self.name}'")
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check error for pool '{self.name}': {e}")
            self._is_healthy = False
            return False
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._pool_lock:
            while self._pool:
                connection = self._pool.pop()
                try:
                    self.close_connection(connection)
                except Exception as e:
                    logger.debug(f"Error closing pooled connection: {e}")
            
            logger.info(f"Closed all connections for pool '{self.name}'")


class RetryManager:
    """Manages retry logic with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig):
        """Initialize retry manager with configuration."""
        self.config = config
    
    def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str,
        retryable_exceptions: tuple = (Exception,)
    ) -> T:
        """
        Execute operation with retry logic.
        
        Args:
            operation: Function to execute
            operation_name: Name for logging
            retryable_exceptions: Tuple of exceptions that should trigger retry
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    logger.debug(f"Retry attempt {attempt} for {operation_name}")
                
                result = operation()
                
                if attempt > 0:
                    logger.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Operation {operation_name} failed on attempt {attempt + 1}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Operation {operation_name} failed after {self.config.max_retries + 1} attempts")
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Operation {operation_name} failed with no exception recorded")
    
    async def execute_with_retry_async(
        self,
        operation: Callable[[], T],
        operation_name: str,
        retryable_exceptions: tuple = (Exception,)
    ) -> T:
        """Async version of execute_with_retry."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    logger.debug(f"Async retry attempt {attempt} for {operation_name}")
                
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    result = operation()
                
                if attempt > 0:
                    logger.info(f"Async operation {operation_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Async operation {operation_name} failed on attempt {attempt + 1}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Async operation {operation_name} failed after {self.config.max_retries + 1} attempts")
        
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Async operation {operation_name} failed with no exception recorded")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            # Add ±25% jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)  # Minimum 0.1s delay


class GracefulDegradationManager:
    """Manages graceful degradation when components are unavailable."""
    
    def __init__(self):
        """Initialize graceful degradation manager."""
        self._component_status: Dict[str, bool] = {}
        self._status_lock = threading.Lock()
        self._fallback_handlers: Dict[str, Callable] = {}
    
    def register_component(self, name: str, health_check: Callable[[], bool]) -> None:
        """
        Register a component for health monitoring.
        
        Args:
            name: Component name
            health_check: Function that returns True if component is healthy
        """
        with self._status_lock:
            self._component_status[name] = True
        
        logger.info(f"Registered component '{name}' for health monitoring")
    
    def register_fallback(self, component_name: str, fallback_handler: Callable) -> None:
        """
        Register a fallback handler for a component.
        
        Args:
            component_name: Name of the component
            fallback_handler: Function to call when component is unavailable
        """
        self._fallback_handlers[component_name] = fallback_handler
        logger.info(f"Registered fallback handler for component '{component_name}'")
    
    def is_component_healthy(self, name: str) -> bool:
        """Check if a component is healthy."""
        with self._status_lock:
            return self._component_status.get(name, False)
    
    def mark_component_unhealthy(self, name: str) -> None:
        """Mark a component as unhealthy."""
        with self._status_lock:
            self._component_status[name] = False
        logger.warning(f"Component '{name}' marked as unhealthy")
    
    def mark_component_healthy(self, name: str) -> None:
        """Mark a component as healthy."""
        with self._status_lock:
            self._component_status[name] = True
        logger.info(f"Component '{name}' marked as healthy")
    
    def execute_with_fallback(
        self,
        component_name: str,
        primary_operation: Callable[[], T],
        operation_name: str
    ) -> Optional[T]:
        """
        Execute operation with fallback if component is unhealthy.
        
        Args:
            component_name: Name of the component
            primary_operation: Primary operation to execute
            operation_name: Name for logging
            
        Returns:
            Result of primary operation or fallback, None if both fail
        """
        if self.is_component_healthy(component_name):
            try:
                return primary_operation()
            except Exception as e:
                logger.error(f"Primary operation {operation_name} failed: {e}")
                self.mark_component_unhealthy(component_name)
        
        # Try fallback
        fallback_handler = self._fallback_handlers.get(component_name)
        if fallback_handler:
            try:
                logger.info(f"Using fallback for {operation_name} (component: {component_name})")
                return fallback_handler()
            except Exception as e:
                logger.error(f"Fallback for {operation_name} failed: {e}")
        else:
            logger.warning(f"No fallback registered for component '{component_name}'")
        
        return None


# Global instances
_connection_pools: Dict[str, ConnectionPool] = {}
_retry_manager = RetryManager(RetryConfig())
_degradation_manager = GracefulDegradationManager()


def get_connection_pool(name: str) -> Optional[ConnectionPool]:
    """Get a connection pool by name."""
    return _connection_pools.get(name)


def register_connection_pool(
    name: str,
    create_connection: Callable[[], Any],
    validate_connection: Callable[[Any], bool],
    close_connection: Callable[[Any], None],
    config: Optional[ConnectionConfig] = None
) -> ConnectionPool:
    """
    Register a new connection pool.
    
    Args:
        name: Pool name
        create_connection: Function to create connections
        validate_connection: Function to validate connections
        close_connection: Function to close connections
        config: Pool configuration
        
    Returns:
        Created connection pool
    """
    if config is None:
        config = ConnectionConfig()
    
    pool = ConnectionPool(name, create_connection, validate_connection, close_connection, config)
    _connection_pools[name] = pool
    
    logger.info(f"Registered connection pool '{name}'")
    return pool


def get_retry_manager() -> RetryManager:
    """Get the global retry manager."""
    return _retry_manager


def get_degradation_manager() -> GracefulDegradationManager:
    """Get the global graceful degradation manager."""
    return _degradation_manager


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,)
):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay between retries
        retryable_exceptions: Exceptions that should trigger retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_config = RetryConfig(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay
            )
            retry_manager = RetryManager(retry_config)
            
            return retry_manager.execute_with_retry(
                lambda: func(*args, **kwargs),
                func.__name__,
                retryable_exceptions
            )
        return wrapper
    return decorator


def with_timeout(timeout: float):
    """
    Decorator for adding timeout to synchronous functions using signal-based alarm (Unix only).

    Args:
        timeout: Timeout in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal

            def _handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {timeout}s")

            try:
                old_handler = signal.signal(signal.SIGALRM, _handler)
                signal.alarm(int(timeout))
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                # SIGALRM not available (Windows), run without timeout
                return func(*args, **kwargs)
        return wrapper
    return decorator


def cleanup_all_pools():
    """Close all connection pools."""
    for name, pool in _connection_pools.items():
        try:
            pool.close_all()
        except Exception as e:
            logger.error(f"Error closing pool '{name}': {e}")
    
    _connection_pools.clear()
    logger.info("Closed all connection pools")