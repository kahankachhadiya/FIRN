"""
Clear All Databases Script

This script clears all data from Qdrant, Elasticsearch, and FalkorDB databases.
Useful for testing with fresh data or resetting the system.

Usage:
    python scripts/clear_all_databases.py [--force]
    
Options:
    --force: Skip confirmation prompt
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from config.production import initialize_production_config
from utils.logging_utils import get_logger
from db.qdrant_client import _get_client as get_qdrant_client
from db.elasticsearch_client import _get_client as get_elasticsearch_client
from db.falkor_client import _get_client as get_falkor_client, _get_knowledge_graph


logger = get_logger(__name__)

# Initialize production configuration
initialize_production_config()


class DatabaseCleaner:
    """Clears all data from Qdrant, Elasticsearch, and FalkorDB."""
    
    def __init__(self):
        self.stats = {
            'qdrant_cleared': False,
            'elasticsearch_cleared': False,
            'falkordb_cleared': False,
            'errors': []
        }
    
    def clear_all_databases(self, force: bool = False):
        """Clear all databases."""
        logger.info("=" * 60)
        logger.info("DATABASE CLEANUP - CLEAR ALL DATA")
        logger.info("=" * 60)
        
        if not force:
            logger.warning("⚠️  This will DELETE ALL DATA from:")
            logger.warning("   - Qdrant (all collections)")
            logger.warning("   - Elasticsearch (all indices)")
            logger.warning("   - FalkorDB (knowledge graph)")
            logger.warning("")
            response = input("Are you sure you want to continue? (yes/N): ")
            if response.lower() != 'yes':
                logger.info("Operation cancelled by user")
                return
        
        logger.info("\nStarting database cleanup...")
        
        # Clear each database
        self._clear_qdrant()
        self._clear_elasticsearch()
        self._clear_falkordb()
        
        # Print summary
        self._print_summary()
    
    def _clear_qdrant(self):
        """Clear all collections from Qdrant."""
        try:
            logger.info("\n[1/3] Clearing Qdrant...")
            client = get_qdrant_client()
            
            # Get all collections
            collections = client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if not collection_names:
                logger.info("  ✓ No collections found in Qdrant")
                self.stats['qdrant_cleared'] = True
                return
            
            # Delete each collection
            for collection_name in collection_names:
                try:
                    client.delete_collection(collection_name)
                    logger.info(f"  ✓ Deleted collection: {collection_name}")
                except Exception as e:
                    error_msg = f"Failed to delete Qdrant collection {collection_name}: {e}"
                    logger.error(f"  ✗ {error_msg}")
                    self.stats['errors'].append(error_msg)
            
            self.stats['qdrant_cleared'] = True
            logger.info(f"  ✓ Cleared {len(collection_names)} Qdrant collection(s)")
            
        except Exception as e:
            error_msg = f"Failed to clear Qdrant: {e}"
            logger.error(f"  ✗ {error_msg}")
            self.stats['errors'].append(error_msg)
    
    def _clear_elasticsearch(self):
        """Clear Elasticsearch indices."""
        try:
            logger.info("\n[2/3] Clearing Elasticsearch...")
            client = get_elasticsearch_client()
            
            # Get all indices
            indices = client.indices.get(index='*')
            index_names = [name for name in indices.keys() if not name.startswith('.')]
            
            if not index_names:
                logger.info("  ✓ No indices found in Elasticsearch")
                self.stats['elasticsearch_cleared'] = True
                return
            
            # Delete each index
            for index_name in index_names:
                try:
                    # Get document count before deleting
                    count_result = client.count(index=index_name)
                    doc_count = count_result['count']
                    
                    client.indices.delete(index=index_name)
                    logger.info(f"  ✓ Deleted index: {index_name} ({doc_count} documents)")
                except Exception as e:
                    error_msg = f"Failed to delete Elasticsearch index {index_name}: {e}"
                    logger.error(f"  ✗ {error_msg}")
                    self.stats['errors'].append(error_msg)
            
            self.stats['elasticsearch_cleared'] = True
            logger.info(f"  ✓ Cleared {len(index_names)} Elasticsearch index/indices")
            
        except Exception as e:
            error_msg = f"Failed to clear Elasticsearch: {e}"
            logger.error(f"  ✗ {error_msg}")
            self.stats['errors'].append(error_msg)
    
    def _clear_falkordb(self):
        """Clear FalkorDB knowledge graph."""
        try:
            logger.info("\n[3/3] Clearing FalkorDB knowledge graph...")
            graph = _get_knowledge_graph()
            
            # Count nodes before clearing
            count_query = "MATCH (n) RETURN count(n) AS node_count"
            result = graph.query(count_query)
            node_count = result.result_set[0][0] if result.result_set else 0
            
            if node_count == 0:
                logger.info("  ✓ FalkorDB knowledge graph is already empty")
                self.stats['falkordb_cleared'] = True
                return
            
            # Delete all nodes and edges
            graph.query("MATCH (n) DETACH DELETE n")
            logger.info(f"  ✓ Cleared {node_count} node(s) from FalkorDB knowledge graph ({settings.falkor_knowledge_graph_name})")
            self.stats['falkordb_cleared'] = True
            
        except Exception as e:
            error_msg = f"Failed to clear FalkorDB: {e}"
            logger.error(f"  ✗ {error_msg}")
            self.stats['errors'].append(error_msg)
    
    def _print_summary(self):
        """Print cleanup summary."""
        logger.info("\n" + "=" * 60)
        logger.info("CLEANUP SUMMARY")
        logger.info("=" * 60)
        
        # Success status
        logger.info("\nStatus:")
        logger.info(f"  Qdrant:         {'✓ Cleared' if self.stats['qdrant_cleared'] else '✗ Failed'}")
        logger.info(f"  Elasticsearch:  {'✓ Cleared' if self.stats['elasticsearch_cleared'] else '✗ Failed'}")
        logger.info(f"  FalkorDB:       {'✓ Cleared' if self.stats['falkordb_cleared'] else '✗ Failed'}")
        
        # Errors
        if self.stats['errors']:
            logger.info(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats['errors']:
                logger.error(f"  - {error}")
        else:
            logger.info("\n✓ All databases cleared successfully!")
        
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Clear all data from Qdrant, Elasticsearch, and FalkorDB")
    parser.add_argument("--force", action="store_true",
                       help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    cleaner = DatabaseCleaner()
    cleaner.clear_all_databases(force=args.force)


if __name__ == "__main__":
    main()
