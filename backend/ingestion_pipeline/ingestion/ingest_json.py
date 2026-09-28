"""
Main ingestion pipeline for processing document JSON files.

This module orchestrates the complete ingestion workflow:
1. Load and validate JSON against DocumentInput schema
2. Generate stable document_id and chunk_ids
3. Write data to all three databases (Redis, Qdrant, Falkor DB)
4. Build graph relations between semantically similar chunks
5. Store document metadata with chunk sequence
"""

from datetime import datetime
from typing import List
from pathlib import Path
import time
import os
import gc

from ingestion.schemas import DocumentInput
from ingestion.id_generator import generate_document_id, generate_chunk_id
from db.elasticsearch_client import save_paragraph, save_document_metadata
from db.qdrant_client import add_paragraph_embedding
from db.falkor_client import add_chunk_vertex, ensure_graph_exists
from utils.json_utils import load_json_file, validate_json_schema
from utils.errors import IngestionError, ValidationError, DatabaseError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def ingest_document_from_json(json_path: str) -> None:
    """
    Ingest a document from a JSON file into the RAG system.
    
    This function performs the complete ingestion workflow:
    1. Load and validate JSON file
    2. Generate document_id from metadata
    3. Generate chunk_ids for all chunks
    4. For each chunk:
       - Write paragraph text to Redis
       - Write paragraph embedding to Qdrant main collection
       - Create vertex in Falkor DB
       - Build graph relations (search similar summaries, create edges)
    6. Store document metadata with chunk_sequence in Redis
    
    Args:
        json_path: Path to the JSON file to ingest
        
    Raises:
        IngestionError: If JSON loading, validation, or chunk size validation fails
        ValidationError: If Pydantic schema validation fails
        DatabaseError: If any database operation fails
    """
    start_time = time.time()
    filename = Path(json_path).name
    
    logger.info("=" * 80)
    logger.info(f"STARTING DOCUMENT INGESTION: {filename}")
    logger.info("=" * 80)
    logger.info(f"JSON file: {json_path}")
    
    # Validate file exists and get size
    if not os.path.exists(json_path):
        error_msg = f"JSON file not found: {json_path}"
        logger.error(error_msg)
        raise IngestionError(error_msg)
    
    file_size = os.path.getsize(json_path)
    logger.info(f"File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    try:
        # Step 1: Load and validate JSON
        logger.info("Stage 1/6: Loading and validating JSON file...")
        load_start = time.time()
        
        json_data = load_json_file(json_path)
        logger.info(f"JSON loaded successfully in {time.time() - load_start:.3f}s")
        
        # Validate against schema
        validate_start = time.time()
        document_input = validate_json_schema(json_data, DocumentInput)
        logger.info(f"JSON validation completed in {time.time() - validate_start:.3f}s")
        logger.info(f"Document title: {document_input.document_metadata.title}")
        logger.info(f"Number of chunks: {len(document_input.chunks)}")
        logger.info(f"Total pages: {document_input.document_metadata.total_pages}")
        
        # Step 2: Generate document ID
        logger.info("Stage 2/6: Generating document ID...")
        id_start = time.time()
        
        document_id = generate_document_id(document_input.document_metadata)
        logger.info(f"Generated document_id: {document_id}")
        logger.info(f"Document ID generation completed in {time.time() - id_start:.3f}s")
        
        # Step 3: Generate chunk IDs
        logger.info("Stage 3/5: Generating chunk IDs...")
        chunk_id_start = time.time()
        
        chunk_ids: List[str] = []
        for chunk in document_input.chunks:
            chunk_id = generate_chunk_id(
                document_id,
                chunk.page_number,
                chunk.chunk_index
            )
            chunk_ids.append(chunk_id)
        
        logger.info(f"Generated {len(chunk_ids)} chunk IDs in {time.time() - chunk_id_start:.3f}s")
        
        # Initialize integration manager for enhanced functionality
        from ingestion.integration_manager import get_integration_manager
        integration_manager = get_integration_manager()
        
        # Ensure database collections exist (handled by integration manager)
        logger.info("Ensuring database collections exist...")
        db_setup_start = time.time()
        ensure_graph_exists()
        logger.info(f"Database setup completed in {time.time() - db_setup_start:.3f}s")
        
        # Step 5: Process each chunk
        logger.info("Stage 4/5: Processing chunks and storing in databases...")
        chunk_processing_start = time.time()
        
        redis_operations = 0
        qdrant_operations = 0
        falkor_operations = 0
        graph_relations = 0
        similar_edges_total = 0
        probe_edges_total = 0
        graph_a_failures = 0
        graph_b_failures = 0
        
        # Progress logging interval (log every N chunks)
        progress_interval = 10
        
        for i, (chunk, chunk_id) in enumerate(zip(document_input.chunks, chunk_ids)):
            chunk_start_time = time.time()
            
            # Log progress every N chunks or for first/last chunk
            if i == 0 or i == len(document_input.chunks) - 1 or (i + 1) % progress_interval == 0:
                elapsed = time.time() - chunk_processing_start
                chunks_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                eta_seconds = (len(document_input.chunks) - (i + 1)) / chunks_per_sec if chunks_per_sec > 0 else 0
                
                logger.info(
                    f"Progress: {i+1}/{len(document_input.chunks)} chunks ({(i+1)/len(document_input.chunks)*100:.1f}%) | "
                    f"Speed: {chunks_per_sec:.1f} chunks/s | "
                    f"ETA: {eta_seconds:.0f}s | "
                    f"Graph A edges: {similar_edges_total} | "
                    f"Graph B edges: {probe_edges_total}"
                )
            else:
                logger.debug(
                    f"Processing chunk {i+1}/{len(document_input.chunks)}: {chunk_id} "
                    f"(page={chunk.page_number}, index={chunk.chunk_index})"
                )
            
            try:
                # Write to Redis (keypair storage)
                redis_start = time.time()
                save_paragraph(chunk_id, chunk.paragraph_text)
                redis_time = time.time() - redis_start
                redis_operations += 1
                logger.debug(f"  ✓ Redis storage: {redis_time:.3f}s")
                
                # Write to Qdrant (vector storage)
                qdrant_start = time.time()
                metadata = {
                    "document_id": document_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index
                }
                add_paragraph_embedding(chunk_id, chunk.paragraph_text, metadata)
                qdrant_time = time.time() - qdrant_start
                qdrant_operations += 1
                logger.debug(f"  ✓ Qdrant embedding: {qdrant_time:.3f}s")
                
                # Write to Falkor DB (graph vertex)
                falkor_start = time.time()
                add_chunk_vertex(chunk_id, document_id, chunk.page_number)
                falkor_time = time.time() - falkor_start
                falkor_operations += 1
                logger.debug(f"  ✓ FalkorDB vertex: {falkor_time:.3f}s")
                
                # Build graph relations using integration manager
                relations_start = time.time()
                edge_results = integration_manager.process_ingestion(
                    chunk_id=chunk_id,
                    summary=chunk.paragraph_text,  # Use paragraph_text as summary since summary field removed
                    document_id=document_id,
                    metadata=metadata,
                    chunk_text=chunk.paragraph_text
                )
                relations_time = time.time() - relations_start
                
                # Extract statistics
                similar_edges = edge_results.get('similar_edges', 0)
                probe_edges = edge_results.get('probe_edges', {})
                total_edges = edge_results.get('total_edges', 0)
                
                # Track failures
                if edge_results.get('graph_a_failed', False):
                    graph_a_failures += 1
                if edge_results.get('graph_b_failed', False):
                    graph_b_failures += 1
                
                graph_relations += total_edges
                similar_edges_total += similar_edges
                probe_edges_total += sum(probe_edges.values())
                
                logger.debug(
                    f"  ✓ Parallel edge creation: {total_edges} edges "
                    f"(similar={similar_edges}, probe={sum(probe_edges.values())}) "
                    f"in {relations_time:.3f}s"
                )
                
                chunk_total_time = time.time() - chunk_start_time
                logger.debug(f"  Chunk {i+1} completed in {chunk_total_time:.3f}s")
                
                # Periodic garbage collection to prevent memory accumulation
                if (i + 1) % 10 == 0:
                    gc.collect()
                    logger.debug(f"Garbage collection triggered after chunk {i+1}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process chunk {i+1}: {str(e)}")
                raise DatabaseError(f"Failed to process chunk {i+1}: {str(e)}") from e
        
        chunk_processing_time = time.time() - chunk_processing_start
        logger.info(f"Chunk processing completed in {chunk_processing_time:.2f}s")
        logger.info(f"Database operations: Redis={redis_operations}, Qdrant={qdrant_operations}, FalkorDB={falkor_operations}")
        logger.info(f"Edge creation: Similar={similar_edges_total}, Probe={probe_edges_total}, Total={graph_relations}")
        
        # Log detailed Graph A Builder statistics
        if (hasattr(integration_manager, 'components') and 
            hasattr(integration_manager.components, 'parallel_edge_creator') and 
            integration_manager.components.parallel_edge_creator is not None and
            hasattr(integration_manager.components.parallel_edge_creator, 'graph_a_builder')):
            graph_a_stats = integration_manager.components.parallel_edge_creator.graph_a_builder.get_statistics()
            logger.info("=" * 80)
            logger.info("Graph A Builder Statistics:")
            logger.info(f"  Chunks processed: {graph_a_stats.get('chunks_processed', 0)}")
            logger.info(f"  Total candidates: {graph_a_stats.get('total_candidates', 0)}")
            logger.info(f"  Candidates after filtering: {graph_a_stats.get('candidates_after_filtering', 0)}")
            logger.info(f"  Edges created: {graph_a_stats.get('edges_created', 0)}")
            logger.info(f"  Avg candidates per chunk: {graph_a_stats.get('avg_candidates_per_chunk', 0):.1f}")
            logger.info(f"  Avg edges per chunk: {graph_a_stats.get('avg_edges_per_chunk', 0):.1f}")
            logger.info(f"  Reranker calls: {graph_a_stats.get('reranker_calls', 0)}")
            logger.info(f"  Reranker failures: {graph_a_stats.get('reranker_failures', 0)}")
            logger.info(f"  Reranker success rate: {graph_a_stats.get('reranker_success_rate', 0):.1%}")
            logger.info("=" * 80)
        
        # Log failure statistics if any failures occurred
        if graph_a_failures > 0 or graph_b_failures > 0:
            logger.warning(
                f"Graph edge creation failures: Graph A={graph_a_failures}, Graph B={graph_b_failures} "
                f"out of {len(document_input.chunks)} chunks"
            )
            if graph_a_failures > len(document_input.chunks) * 0.1:
                logger.error(
                    f"Graph A failure rate ({graph_a_failures/len(document_input.chunks):.1%}) exceeds 10% threshold. "
                    f"Check reranker service health and database connectivity."
                )
            if graph_b_failures > len(document_input.chunks) * 0.1:
                logger.error(
                    f"Graph B failure rate ({graph_b_failures/len(document_input.chunks):.1%}) exceeds 10% threshold. "
                    f"Check probe graph manager and classification service health."
                )
        
        # Step 6: Store document metadata
        logger.info("Stage 5/5: Storing document metadata...")
        metadata_start = time.time()
        
        # Generate ingestion_timestamp at pipeline execution time
        ingestion_timestamp = datetime.utcnow().isoformat()
        
        metadata_dict = {
            "title": document_input.document_metadata.title,
            "path": document_input.document_metadata.path,
            "author": document_input.document_metadata.author,
            "created_date": document_input.document_metadata.created_date,
            "source_type": document_input.document_metadata.source_type,
            "file_name": document_input.document_metadata.file_name,
            "language": document_input.document_metadata.language,
            "total_pages": document_input.document_metadata.total_pages,
            "tags": document_input.document_metadata.tags,
            "chunk_sequence": chunk_ids,
            "ingestion_timestamp": ingestion_timestamp,
            "extra_metadata": document_input.document_metadata.extra_metadata
        }
        
        save_document_metadata(document_id, metadata_dict)
        
        metadata_time = time.time() - metadata_start
        logger.info(f"Document metadata stored in {metadata_time:.3f}s")
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✓ DOCUMENT INGESTION COMPLETED: {filename}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Document ID: {document_id}")
        logger.info(f"Chunks processed: {len(document_input.chunks)}")
        logger.info(f"Performance: {len(document_input.chunks) / total_time:.1f} chunks/second")
        logger.info("Document is now available for RAG queries")
        logger.info("=" * 80)
        
    except (IngestionError, ValidationError, DatabaseError):
        # Re-raise our custom exceptions as-is
        total_time = time.time() - start_time
        logger.warning(f"✗ Document ingestion failed: {filename} (after {total_time:.2f}s)")
        raise
    except Exception as e:
        # Wrap unexpected exceptions in IngestionError
        total_time = time.time() - start_time
        logger.warning(f"✗ Document ingestion failed: {filename} (unexpected error after {total_time:.2f}s): {str(e)}")
        logger.exception("Stack trace:")
        raise IngestionError(f"Unexpected error during ingestion: {e}") from e
