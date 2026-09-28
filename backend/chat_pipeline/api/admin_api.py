"""
Admin API for Database Management

This module provides REST API endpoints for administrative database operations
including clearing all data from Qdrant, Elasticsearch, and FalkorDB.

Endpoints:
- POST /admin/clear-databases - Clear all data from all databases
- GET /admin/database-stats - Get statistics about database contents
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from db.qdrant_client import _get_client as get_qdrant_client
from db.elasticsearch_client import _get_client as get_elasticsearch_client
from db.falkor_client import _get_client as get_falkor_client, _get_knowledge_graph
from config.settings import settings


logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class ClearDatabasesRequest(BaseModel):
    """Request model for clearing databases."""
    confirm: bool = Field(..., description="Must be true to confirm the operation")
    databases: list[str] = Field(
        default=["qdrant", "elasticsearch", "falkordb"],
        description="List of databases to clear (qdrant, elasticsearch, falkordb)"
    )


class ClearDatabasesResponse(BaseModel):
    """Response model for clearing databases."""
    success: bool
    message: str
    cleared: Dict[str, bool]
    errors: list[str] = []


class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics."""
    qdrant: Dict[str, Any]
    elasticsearch: Dict[str, Any]
    falkordb: Dict[str, Any]


# Create API router
admin_router = APIRouter(prefix="/admin", tags=["Admin"])


@admin_router.post("/clear-databases", response_model=ClearDatabasesResponse)
async def clear_databases(request: ClearDatabasesRequest):
    """
    Clear all data from specified databases (Qdrant, Elasticsearch, FalkorDB).
    
    WARNING: This operation is irreversible and will delete all data!
    
    Args:
        request: Clear databases request with confirmation
        
    Returns:
        ClearDatabasesResponse: Result of the clear operation
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set 'confirm' to true to proceed with database clearing"
        )
    
    cleared = {
        "qdrant": False,
        "elasticsearch": False,
        "falkordb": False
    }
    errors = []
    
    # Clear Qdrant
    if "qdrant" in request.databases:
        try:
            logger.info("Clearing Qdrant collections...")
            client = get_qdrant_client()
            
            # Get all collections
            collections = client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            # Delete each collection
            for collection_name in collection_names:
                try:
                    client.delete_collection(collection_name)
                    logger.info(f"Deleted Qdrant collection: {collection_name}")
                except Exception as e:
                    error_msg = f"Failed to delete Qdrant collection {collection_name}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            cleared["qdrant"] = True
            logger.info(f"Cleared {len(collection_names)} Qdrant collection(s)")
            
        except Exception as e:
            error_msg = f"Failed to clear Qdrant: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Clear Elasticsearch
    if "elasticsearch" in request.databases:
        try:
            logger.info("Clearing Elasticsearch indices...")
            client = get_elasticsearch_client()
            
            # Get index names from settings
            chunk_index = settings.elasticsearch_chunk_index
            metadata_index = settings.elasticsearch_metadata_index
            
            indices_to_clear = [chunk_index, metadata_index]
            cleared_count = 0
            
            # Delete each index
            for index_name in indices_to_clear:
                try:
                    if client.indices.exists(index=index_name):
                        # Get document count before deletion
                        count_result = client.count(index=index_name)
                        doc_count = count_result.get('count', 0)
                        
                        # Delete the index
                        client.indices.delete(index=index_name)
                        logger.info(f"Deleted Elasticsearch index: {index_name} ({doc_count} documents)")
                        cleared_count += 1
                    else:
                        logger.info(f"Elasticsearch index {index_name} does not exist, skipping")
                except Exception as e:
                    error_msg = f"Failed to delete Elasticsearch index {index_name}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            cleared["elasticsearch"] = cleared_count > 0
            logger.info(f"Cleared {cleared_count} Elasticsearch index/indices")
            
        except Exception as e:
            error_msg = f"Failed to clear Elasticsearch: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Clear FalkorDB
    if "falkordb" in request.databases:
        try:
            logger.info("Clearing FalkorDB knowledge graph...")
            graph = _get_knowledge_graph()
            
            # Count nodes before clearing
            count_query = "MATCH (n) RETURN count(n) AS node_count"
            result = graph.query(count_query)
            node_count = result.result_set[0][0] if result.result_set else 0
            
            # Delete all nodes and edges
            graph.query("MATCH (n) DETACH DELETE n")
            cleared["falkordb"] = True
            logger.info(f"Cleared {node_count} node(s) from FalkorDB knowledge graph ({settings.falkor_knowledge_graph_name})")
            
        except Exception as e:
            error_msg = f"Failed to clear FalkorDB: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Determine overall success
    success = all(cleared.values()) and len(errors) == 0
    
    if success:
        message = "All databases cleared successfully"
    elif any(cleared.values()):
        message = "Some databases cleared successfully, but some operations failed"
    else:
        message = "Failed to clear databases"
    
    return ClearDatabasesResponse(
        success=success,
        message=message,
        cleared=cleared,
        errors=errors
    )


@admin_router.get("/database-stats", response_model=DatabaseStatsResponse)
async def get_database_stats():
    """
    Get statistics about database contents.
    
    Returns:
        DatabaseStatsResponse: Statistics for each database
    """
    stats = {
        "qdrant": {},
        "elasticsearch": {},
        "falkordb": {}
    }
    
    # Qdrant stats
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        
        qdrant_stats = {
            "collections": []
        }
        
        for col in collections:
            try:
                collection_info = client.get_collection(col.name)
                qdrant_stats["collections"].append({
                    "name": col.name,
                    "vectors_count": collection_info.vectors_count,
                    "points_count": collection_info.points_count
                })
            except Exception as e:
                logger.error(f"Error getting Qdrant collection info for {col.name}: {e}")
        
        stats["qdrant"] = qdrant_stats
        
    except Exception as e:
        logger.error(f"Error getting Qdrant stats: {e}")
        stats["qdrant"] = {"error": str(e)}
    
    # Elasticsearch stats
    try:
        client = get_elasticsearch_client()
        
        # Get index names from settings
        chunk_index = settings.elasticsearch_chunk_index
        metadata_index = settings.elasticsearch_metadata_index
        
        elasticsearch_stats = {
            "indices": []
        }
        
        for index_name in [chunk_index, metadata_index]:
            try:
                if client.indices.exists(index=index_name):
                    # Get document count
                    count_result = client.count(index=index_name)
                    doc_count = count_result.get('count', 0)
                    
                    # Get index stats
                    index_stats = client.indices.stats(index=index_name)
                    size_in_bytes = index_stats['indices'][index_name]['total']['store']['size_in_bytes']
                    
                    elasticsearch_stats["indices"].append({
                        "name": index_name,
                        "documents_count": doc_count,
                        "size_in_bytes": size_in_bytes
                    })
                else:
                    elasticsearch_stats["indices"].append({
                        "name": index_name,
                        "documents_count": 0,
                        "size_in_bytes": 0,
                        "exists": False
                    })
            except Exception as e:
                logger.error(f"Error getting Elasticsearch index info for {index_name}: {e}")
                elasticsearch_stats["indices"].append({
                    "name": index_name,
                    "error": str(e)
                })
        
        stats["elasticsearch"] = elasticsearch_stats
        
    except Exception as e:
        logger.error(f"Error getting Elasticsearch stats: {e}")
        stats["elasticsearch"] = {"error": str(e)}
    
    # FalkorDB stats
    try:
        graph = _get_knowledge_graph()
        
        # Count nodes
        count_query = "MATCH (n) RETURN count(n) AS node_count"
        result = graph.query(count_query)
        node_count = result.result_set[0][0] if result.result_set else 0
        
        # Count edges
        edge_query = "MATCH ()-[r]->() RETURN count(r) AS edge_count"
        result = graph.query(edge_query)
        edge_count = result.result_set[0][0] if result.result_set else 0
        
        falkordb_stats = {
            "graph_name": settings.falkor_knowledge_graph_name,
            "nodes_count": node_count,
            "edges_count": edge_count
        }
        
        stats["falkordb"] = falkordb_stats
        
    except Exception as e:
        logger.error(f"Error getting FalkorDB stats: {e}")
        stats["falkordb"] = {"error": str(e)}
    
    return DatabaseStatsResponse(**stats)


@admin_router.get("/health")
async def admin_health_check():
    """
    Simple health check for admin API.
    
    Returns:
        Dict: Health status
    """
    return {
        "status": "healthy",
        "message": "Admin API is operational"
    }
