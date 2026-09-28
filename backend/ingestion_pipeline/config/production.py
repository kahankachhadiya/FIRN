"""
Production configuration module with environment-based configuration loading.

This module provides production-ready configuration classes that load all
settings from environment variables with proper validation, defaults, and
connection pooling parameters for the layered graph RAG system.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging
import json
from pathlib import Path
from datetime import datetime

# Load environment variables from the pipeline's .env file
try:
    from dotenv import load_dotenv
    _pipeline_env = Path(__file__).parent.parent / '.env'
    if _pipeline_env.exists():
        load_dotenv(_pipeline_env)
        logging.getLogger(__name__).info(f"Loaded configuration from {_pipeline_env}")
    else:
        logging.getLogger(__name__).warning(f"No .env file found at {_pipeline_env}")
except ImportError:
    logging.getLogger(__name__).warning("python-dotenv not available, environment variables must be set manually")


@dataclass
class BackupIntegrationConfig:
    """Backup system integration configuration."""
    
    # Basic backup settings
    enabled: bool = False
    backup_dir: str = "./backups"
    interval_hours: int = 24
    retention_days: int = 7
    
    # Component selection
    redis_backup_enabled: bool = True
    qdrant_backup_enabled: bool = True
    falkordb_backup_enabled: bool = True
    logs_backup_enabled: bool = True
    config_backup_enabled: bool = True
    
    # Performance settings
    compression_enabled: bool = True
    verify_backups: bool = True
    
    @classmethod
    def from_env(cls) -> 'BackupIntegrationConfig':
        """Load backup integration config from environment variables."""
        return cls(
            enabled=os.getenv('BACKUP_ENABLED', 'false').lower() == 'true',
            backup_dir=os.getenv('BACKUP_DIR', './backups'),
            interval_hours=int(os.getenv('BACKUP_INTERVAL_HOURS', '24')),
            retention_days=int(os.getenv('BACKUP_RETENTION_DAYS', '7')),
            redis_backup_enabled=os.getenv('BACKUP_REDIS_ENABLED', 'true').lower() == 'true',
            qdrant_backup_enabled=os.getenv('BACKUP_QDRANT_ENABLED', 'true').lower() == 'true',
            falkordb_backup_enabled=os.getenv('BACKUP_FALKORDB_ENABLED', 'true').lower() == 'true',
            logs_backup_enabled=os.getenv('BACKUP_LOGS_ENABLED', 'true').lower() == 'true',
            config_backup_enabled=os.getenv('BACKUP_CONFIG_ENABLED', 'true').lower() == 'true',
            compression_enabled=os.getenv('BACKUP_COMPRESSION_ENABLED', 'true').lower() == 'true',
            verify_backups=os.getenv('BACKUP_VERIFY_ENABLED', 'true').lower() == 'true'
        )


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    
    host: str
    port: int
    password: Optional[str] = None
    username: Optional[str] = None
    
    connection_timeout: int = 5
    
    # Retry configuration
    retry_enabled: bool = True
    retry_backoff_multiplier: float = 2.0
    retry_max_delay: int = 30
    
    # Qdrant-specific settings
    https: bool = False
    prefer_grpc: bool = False
    verify_ssl: bool = True
    
    # Qdrant HNSW index configuration
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    indexing_threshold: int = 10000


@dataclass
class RerankerConfig:
    """Reranker service configuration."""
    
    # Service configuration
    service_url: str = "http://localhost:8002"
    batch_size: int = 32
    timeout: float = 30.0


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    
    metrics_file_path: str = "logs/metrics.json"
    log_file_path: str = "logs/production.log"
    conversation_log_path: str = "logs/conversation_debug.jsonl"
    structured_logging_enabled: bool = True


@dataclass
class SecurityConfig:
    """Security and rate limiting configuration."""
    
    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 10
    rate_limit_window: int = 60
    
    # Authentication
    auth_enabled: bool = False
    auth_type: str = "api_key"
    jwt_secret: Optional[str] = None
    jwt_expiry: int = 3600
    
    # IP filtering
    ip_whitelist: Optional[str] = None
    ip_blacklist: Optional[str] = None
    
    # Security headers
    secure_headers: bool = True


@dataclass
class ProductionConfig:
    """
    Production configuration settings loaded from environment variables.
    
    This class aggregates all configuration sections and provides methods
    for loading from environment variables with proper validation and defaults.
    
    All pipeline parameters are configurable via environment variables with
    sensible defaults. No hardcoded values are used in the pipeline logic.
    """
    
    # Configuration sections
    redis_config: DatabaseConfig
    qdrant_config: DatabaseConfig
    falkordb_config: DatabaseConfig
    reranker_config: RerankerConfig
    monitoring_config: MonitoringConfig
    security_config: SecurityConfig
    backup_config: BackupIntegrationConfig
    
    @staticmethod
    def _load_multiline_env_var(var_name: str) -> str:
        """
        Load multi-line environment variable and process escape sequences.
        
        Args:
            var_name: Name of the environment variable
            
        Returns:
            str: The processed environment variable value with newlines
        """
        value = os.getenv(var_name, '')
        if value:
            # Replace \n with actual newlines
            value = value.replace('\\n', '\n')
            # Remove surrounding quotes if present
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
        return value
    
    @staticmethod
    def _get_required_env(var_name: str, error_message: str) -> str:
        """
        Get a required environment variable or raise an error.
        
        Args:
            var_name: Name of the environment variable
            error_message: Error message to display if variable is not set
            
        Returns:
            str: Value of the environment variable
            
        Raises:
            EnvironmentError: If the environment variable is not set
        """
        value = os.getenv(var_name)
        if value is None or value.strip() == '':
            raise EnvironmentError(
                f"Required environment variable '{var_name}' is not set. {error_message}"
            )
        return value.strip()
    
    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        """
        Load configuration from environment variables with validation and defaults.

        All fields use os.getenv with sensible defaults so the app starts even
        when legacy services (Redis) are absent from the deployment.

        Returns:
            ProductionConfig: Fully configured production settings
        """
        logger = logging.getLogger(__name__)

        def _i(key: str, default: int) -> int:
            return int(os.getenv(key, str(default)))

        def _f(key: str, default: float) -> float:
            return float(os.getenv(key, str(default)))

        def _b(key: str, default: bool) -> bool:
            return os.getenv(key, str(default)).lower() == 'true'

        def _s(key: str, default: str = '') -> str:
            return os.getenv(key, default)

        # ── Shared pool defaults ──────────────────────────────────────────────
        retry_enabled     = _b('CONNECTION_RETRY_ENABLED', True)
        retry_backoff     = _f('CONNECTION_RETRY_BACKOFF_MULTIPLIER', 2.0)
        retry_max_delay   = _i('CONNECTION_RETRY_MAX_DELAY', 30)

        # ── Redis (not used in active pipeline — all defaults) ────────────────
        redis_config = DatabaseConfig(
            host=_s('REDIS_HOST', 'localhost'),
            port=_i('REDIS_PORT', 6379),
            password=os.getenv('REDIS_PASSWORD'),
            connection_timeout=_i('REDIS_TIMEOUT', 5),
            retry_enabled=retry_enabled,
            retry_backoff_multiplier=retry_backoff,
            retry_max_delay=retry_max_delay,
        )

        # ── Qdrant ────────────────────────────────────────────────────────────
        qdrant_config = DatabaseConfig(
            host=_s('QDRANT_HOST', 'localhost'),
            port=_i('QDRANT_PORT', 6333),
            password=os.getenv('QDRANT_API_KEY'),
            connection_timeout=_i('QDRANT_TIMEOUT', 10),
            retry_enabled=retry_enabled,
            retry_backoff_multiplier=retry_backoff,
            retry_max_delay=retry_max_delay,
        )
        qdrant_config.https            = _b('QDRANT_HTTPS', False)
        qdrant_config.prefer_grpc      = _b('QDRANT_PREFER_GRPC', False)
        qdrant_config.verify_ssl       = _b('QDRANT_VERIFY_SSL', True)
        qdrant_config.hnsw_m           = _i('QDRANT_HNSW_M', 16)
        qdrant_config.hnsw_ef_construct= _i('QDRANT_HNSW_EF_CONSTRUCT', 100)
        qdrant_config.indexing_threshold=_i('QDRANT_INDEXING_THRESHOLD', 10000)

        # ── FalkorDB ──────────────────────────────────────────────────────────
        falkordb_config = DatabaseConfig(
            host=_s('FALKORDB_HOST', 'localhost'),
            port=_i('FALKORDB_PORT', 6380),
            password=os.getenv('FALKORDB_PASSWORD'),
            username=os.getenv('FALKOR_USERNAME'),
            connection_timeout=_i('FALKORDB_TIMEOUT', 5),
            retry_enabled=retry_enabled,
            retry_backoff_multiplier=retry_backoff,
            retry_max_delay=retry_max_delay,
        )

        # ── Reranker ──────────────────────────────────────────────────────────
        reranker_config = RerankerConfig(
            service_url=_s('RERANKER_SERVICE_URL', 'http://localhost:7997'),
            batch_size=_i('RERANKER_BATCH_SIZE', 10),
            timeout=_f('RERANKER_TIMEOUT', 30.0),
        )

        # ── Monitoring ────────────────────────────────────────────────────────
        monitoring_config = MonitoringConfig(
            metrics_file_path=_s('METRICS_FILE_PATH', 'logs/metrics.json'),
            log_file_path=_s('LOG_FILE_PATH', 'logs/production.log'),
            conversation_log_path=_s('CONVERSATION_LOG_PATH', 'logs/conversation_debug.jsonl'),
            structured_logging_enabled=_b('STRUCTURED_LOGGING_ENABLED', True),
        )

        # ── Security ──────────────────────────────────────────────────────────
        security_config = SecurityConfig(
            rate_limit_per_minute=_i('RATE_LIMIT_PER_MINUTE', 100),
            rate_limit_burst=_i('RATE_LIMIT_BURST', 10),
            rate_limit_window=_i('RATE_LIMIT_WINDOW', 60),
            auth_enabled=_b('AUTH_ENABLED', False),
            auth_type=_s('AUTH_TYPE', 'api_key'),
            jwt_secret=os.getenv('JWT_SECRET'),
            jwt_expiry=_i('JWT_EXPIRY', 3600),
            ip_whitelist=os.getenv('IP_WHITELIST'),
            ip_blacklist=os.getenv('IP_BLACKLIST'),
            secure_headers=_b('SECURE_HEADERS', True),
        )

        # ── Backup ────────────────────────────────────────────────────────────
        backup_config = BackupIntegrationConfig.from_env()

        # ── Assemble ──────────────────────────────────────────────────────────
        config = cls(
            redis_config=redis_config,
            qdrant_config=qdrant_config,
            falkordb_config=falkordb_config,
            reranker_config=reranker_config,
            monitoring_config=monitoring_config,
            security_config=security_config,
            backup_config=backup_config,
        )

        config.validate()
        logger.info("Production configuration loaded successfully from environment variables")
        return config

    def validate(self) -> None:
        """
        Comprehensive configuration validation with detailed error reporting and default handling.

        Raises:
            ValueError: If critical configuration validation fails
        """
        errors = []
        warnings = []
        logger = logging.getLogger(__name__)

        # =============================================================================
        # DATABASE CONFIGURATION VALIDATION
        # =============================================================================

        # Validate Redis configuration
        if not self.redis_config.host:
            errors.append("Redis host is required (REDIS_HOST)")
        if not (1 <= self.redis_config.port <= 65535):
            errors.append(f"Redis port must be between 1 and 65535, got: {self.redis_config.port}")

        # Validate Qdrant configuration
        if not self.qdrant_config.host:
            errors.append("Qdrant host is required (QDRANT_HOST)")
        if not (1 <= self.qdrant_config.port <= 65535):
            errors.append(f"Qdrant port must be between 1 and 65535, got: {self.qdrant_config.port}")
        if self.qdrant_config.connection_timeout <= 0:
            errors.append(f"Qdrant connection_timeout must be positive, got: {self.qdrant_config.connection_timeout}")

        # Validate FalkorDB configuration
        if not self.falkordb_config.host:
            errors.append("FalkorDB host is required (FALKORDB_HOST)")
        if not (1 <= self.falkordb_config.port <= 65535):
            errors.append(f"FalkorDB port must be between 1 and 65535, got: {self.falkordb_config.port}")

        # Validate connection pool settings across all databases
        for db_name, config in [("Redis", self.redis_config), ("Qdrant", self.qdrant_config), ("FalkorDB", self.falkordb_config)]:
            if config.retry_backoff_multiplier <= 0:
                errors.append(f"{db_name} retry_backoff_multiplier must be positive, got: {config.retry_backoff_multiplier}")
            if config.retry_max_delay <= 0:
                errors.append(f"{db_name} retry_max_delay must be positive, got: {config.retry_max_delay}")

        # =============================================================================
        # RERANKER CONFIGURATION VALIDATION
        # =============================================================================
        
        # Validate reranker service URL
        if not self.reranker_config.service_url:
            errors.append("Reranker service URL is required (RERANKER_SERVICE_URL)")
        else:
            import re
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:'
                r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain with TLD
                r'localhost|'  # localhost
                r'[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?|'  # hostname without TLD
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'  # IP address
                r')'
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)?$', re.IGNORECASE)
            
            if not url_pattern.match(self.reranker_config.service_url):
                errors.append(f"reranker_service_url must be a valid URL, got: {self.reranker_config.service_url}")
        
        # Validate reranker batch size
        if self.reranker_config.batch_size <= 0:
            errors.append(f"reranker_batch_size must be positive, got: {self.reranker_config.batch_size}")
        if self.reranker_config.batch_size > 1000:
            warnings.append(f"reranker_batch_size ({self.reranker_config.batch_size}) is very high, may cause memory issues")
        
        # Validate reranker timeout
        if self.reranker_config.timeout <= 0:
            errors.append(f"reranker_timeout must be positive, got: {self.reranker_config.timeout}")
        if self.reranker_config.timeout > 300:
            warnings.append(f"reranker_timeout ({self.reranker_config.timeout}s) is very high, may cause request timeouts")
        
        # =============================================================================
        # MONITORING CONFIGURATION VALIDATION
        # =============================================================================
        
        # Validate log configuration
        if not self.monitoring_config.log_file_path:
            warnings.append("log_file_path not set, using default")
            self.monitoring_config.log_file_path = "logs/production.log"
        if not self.monitoring_config.conversation_log_path:
            warnings.append("conversation_log_path not set, using default")
            self.monitoring_config.conversation_log_path = "logs/conversation_debug.jsonl"
        
        # =============================================================================
        # SECURITY CONFIGURATION VALIDATION
        # =============================================================================
        
        # Validate rate limiting
        if self.security_config.rate_limit_per_minute <= 0:
            errors.append(f"rate_limit_per_minute must be positive, got: {self.security_config.rate_limit_per_minute}")
        if self.security_config.rate_limit_burst < 0:
            errors.append(f"rate_limit_burst must be non-negative, got: {self.security_config.rate_limit_burst}")
        if self.security_config.rate_limit_window <= 0:
            errors.append(f"rate_limit_window must be positive, got: {self.security_config.rate_limit_window}")
        
        # Validate authentication configuration
        if self.security_config.auth_enabled:
            valid_auth_types = ['api_key', 'jwt', 'oauth']
            if self.security_config.auth_type not in valid_auth_types:
                errors.append(f"auth_type must be one of {valid_auth_types}, got: {self.security_config.auth_type}")
            
            if self.security_config.auth_type == 'jwt':
                if not self.security_config.jwt_secret:
                    errors.append("JWT secret is required when auth_type=jwt")
                if self.security_config.jwt_expiry <= 0:
                    errors.append(f"jwt_expiry must be positive, got: {self.security_config.jwt_expiry}")
        
        # =============================================================================
        # APPLICATION CONFIGURATION VALIDATION
        # =============================================================================
        
        # Validate data directories are set
        if not self.monitoring_config.log_file_path:
            warnings.append("log_file_path not set, using default")
        
        # =============================================================================
        # CROSS-CONFIGURATION VALIDATION
        # =============================================================================
        
        # Validate that connection pool sizes are reasonable for worker count
        pass
        
        # =============================================================================
        # FINALIZE VALIDATION
        # =============================================================================
        
        # Log all warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        # Raise error if any critical issues found
        if errors:
            error_message = f"Configuration validation failed with {len(errors)} errors:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_message)
            raise ValueError(error_message)
        
        # Log successful validation
        validation_summary = f"Configuration validation passed with {len(warnings)} warnings"
        if warnings:
            validation_summary += f" (see warnings above)"
        logger.info(validation_summary)
    
    def get_database_urls(self) -> Dict[str, str]:
        """
        Get database connection URLs for easy client initialization.
        
        Returns:
            Dict[str, str]: Database connection URLs
        """
        urls = {}
        
        # Redis URL
        redis_auth = f":{self.redis_config.password}@" if self.redis_config.password else ""
        urls['redis'] = f"redis://{redis_auth}{self.redis_config.host}:{self.redis_config.port}"
        
        # Qdrant URL
        urls['qdrant'] = f"http://{self.qdrant_config.host}:{self.qdrant_config.port}"
        
        # FalkorDB URL
        falkor_auth = f":{self.falkordb_config.password}@" if self.falkordb_config.password else ""
        urls['falkordb'] = f"redis://{falkor_auth}{self.falkordb_config.host}:{self.falkordb_config.port}"
        
        return urls
    
    def create_data_directories(self) -> None:
        """Create log directories if they don't exist."""
        directories = [
            os.path.dirname(self.monitoring_config.log_file_path),
            os.path.dirname(self.monitoring_config.conversation_log_path),
            os.path.dirname(self.monitoring_config.metrics_file_path)
        ]
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration for logging/debugging."""
        return {
            'databases': {
                'redis': f"{self.redis_config.host}:{self.redis_config.port}",
                'qdrant': f"{self.qdrant_config.host}:{self.qdrant_config.port}",
                'falkordb': f"{self.falkordb_config.host}:{self.falkordb_config.port}"
            },
            'application': {},
            'reranker': {
                'service_url': self.reranker_config.service_url,
                'batch_size': self.reranker_config.batch_size,
                'timeout': self.reranker_config.timeout
            },
            'monitoring': {
                'log_file_path': self.monitoring_config.log_file_path
            },
            'security': {
                'auth_enabled': self.security_config.auth_enabled,
                'rate_limit_per_minute': self.security_config.rate_limit_per_minute
            }
        }

    def get_pipeline_parameters(self) -> Dict[str, Any]:
        """Get ingestion pipeline parameters."""
        return {
            'reranker_service_url': self.reranker_config.service_url,
            'reranker_batch_size': self.reranker_config.batch_size,
            'reranker_timeout': self.reranker_config.timeout,
            'classification_batch_size': int(self._get_required_env('CLASSIFICATION_BATCH_SIZE', 'CLASSIFICATION_BATCH_SIZE is required')),
            'classification_service_url': self._get_required_env('CLASSIFICATION_SERVICE_URL', 'CLASSIFICATION_SERVICE_URL is required'),
            'processing_mode': self._get_required_env('PROCESSING_MODE', 'PROCESSING_MODE must be set to "parallel" or "concurrent"'),
            'vector_search_top_k_per_probe': int(self._get_required_env('VECTOR_SEARCH_TOP_K_PER_PROBE', 'VECTOR_SEARCH_TOP_K_PER_PROBE is required')),
            'graph_a_reranker_candidates': int(self._get_required_env('GRAPH_A_RERANKER_CANDIDATES', 'GRAPH_A_RERANKER_CANDIDATES is required')),
        }


# Global production configuration instance
production_config: Optional[ProductionConfig] = None


def get_production_config() -> ProductionConfig:
    """
    Get the global production configuration instance.
    
    Returns:
        ProductionConfig: The global configuration instance
        
    Raises:
        RuntimeError: If configuration has not been initialized
    """
    global production_config
    
    if production_config is None:
        raise RuntimeError("Production configuration has not been initialized. Call initialize_production_config() first.")
    
    return production_config


def initialize_production_config() -> ProductionConfig:
    """
    Initialize the global production configuration from environment variables.
    
    Returns:
        ProductionConfig: The initialized configuration instance
    """
    global production_config
    
    if production_config is None:
        production_config = ProductionConfig.from_env()
    
    return production_config


def reload_production_config() -> ProductionConfig:
    """
    Reload the global production configuration from environment variables.
    
    This function forces a complete reload of the configuration, which should
    only be used for critical configuration changes that require a full restart.
    For non-critical settings, use the hot-reload manager instead.
    
    Returns:
        ProductionConfig: The reloaded configuration instance
    """
    global production_config
    
    production_config = ProductionConfig.from_env()
    return production_config
