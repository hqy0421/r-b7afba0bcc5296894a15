#!/usr/bin/env python
"""
Script to regenerate embeddings for all articles in the RSSidian database.

This script recomputes embedding vectors, stores them in the database, and
publishes a generation-consistent Annoy vector index through the same pipeline
used by ``rssidian rebuild-index``.  If every article already has an up-to-date
vector, it simply re-publishes the index.
"""

import logging
import os
import sys
from typing import List

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer

from rssidian.config import Config
from rssidian.models import Article, get_db_session
from rssidian.vector_index import VectorIndexManager, ensure_schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def generate_embedding(model, text: str) -> List[float]:
    """Generate an embedding vector for text."""
    # Truncate text if too long (most models have a max token limit)
    max_chars = 5000
    truncated_text = text[:max_chars] if len(text) > max_chars else text
    return model.encode(truncated_text).tolist()


def main():
    """Main function to regenerate embeddings."""
    # Load configuration
    config = Config()
    db_session = get_db_session(config.db_path)

    try:
        engine = db_session.get_bind()
        ensure_schema(engine)

        total_articles = db_session.query(Article).count()
        logger.info(f"Found {total_articles} articles in the database")

        # Load embedding model
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer(config.embedding_model_name)

        # Recompute vectors for every article that still has content.
        stats = {"articles_processed": 0, "with_embedding": 0}
        articles = db_session.query(Article).all()
        logger.info(f"Processing {len(articles)} articles...")
        for article in articles:
            stats["articles_processed"] += 1
            text_to_embed = article.content
            if text_to_embed:
                if article.summary:
                    text_to_embed = article.summary + "\n\n" + text_to_embed
                article.embedding_vector = _dump_embedding(
                    generate_embedding(embedding_model, text_to_embed)
                )
                article.embedding_generated = True
                stats["with_embedding"] += 1
            else:
                # No content means no vector: clear stale state so the index
                # and database stay consistent.
                article.embedding_vector = None
                article.embedding_generated = False

        # Commit vectors BEFORE publishing so the published generation is
        # built from persisted data.
        db_session.commit()

        # Publish through the shared, generation-aware pipeline.
        manager = VectorIndexManager(engine, config)
        result = manager.publish()
        if result.get("published"):
            logger.info(
                f"Published vector index generation {result['generation']} "
                f"({result['indexed']} vectors, {result['eligible']} eligible)"
            )
        else:
            logger.error(f"Failed to publish vector index: {result.get('reason')}")

        logger.info(f"Processed {stats['articles_processed']} articles:")
        logger.info(f"  With embeddings: {stats['with_embedding']}")

    except Exception as e:
        logger.error(f"Error during embedding regeneration: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        db_session.close()


def _dump_embedding(embedding: List[float]) -> str:
    import json
    return json.dumps(embedding)


if __name__ == "__main__":
    main()
