"""Tests for generation-aware Annoy vector index behaviour.

These tests avoid downloading/instantiating SentenceTransformer by injecting
deterministic vectors directly and monkeypatching ``_generate_embedding``.
"""

import json
import os
import sqlite3
import sys

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rssidian.models import Article, Base, Feed  # noqa: E402
from rssidian.vector_index import VectorIndexManager, ensure_schema  # noqa: E402

DIM = 8


class FakeConfig:
    """Minimal config object matching the properties used by the code."""

    def __init__(self, tmp_path, metric="angular"):
        self._index_path = str(tmp_path / "annoy.idx")
        self.annoy_n_trees = 4
        self.annoy_metric = metric
        self.embedding_dim = DIM
        self.embedding_model_name = "test-model"
        self.similarity_threshold = 0.65
        self.excerpt_length = 300

    @property
    def annoy_index_path(self):
        return self._index_path


def _unit(index: int):
    vec = [0.0] * DIM
    vec[index % DIM] = 1.0
    return vec


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    ensure_schema(engine)
    session = sessionmaker(bind=engine)()
    feed = Feed(title="Test Feed", url="http://feed.example/rss")
    session.add(feed)
    session.commit()
    yield engine, session, feed.id
    session.close()


def _add_article(session, guid, vector=None, generated=None, content="text", feed_id=1):
    article = Article(
        feed_id=feed_id,
        title=f"Title {guid}",
        url=f"http://x/{guid}",
        guid=guid,
        content=content,
        processed=True,
    )
    session.add(article)
    session.commit()
    if vector is not None:
        article.embedding_vector = json.dumps(vector)
        article.embedding_generated = True
        session.commit()
    elif generated is not None:
        article.embedding_generated = generated
        session.commit()
    return article.id


def _processor(engine, config, session):
    from rssidian.core import RSSProcessor

    proc = RSSProcessor(config, session)
    proc._generate_embedding = lambda text: _unit(0)  # avoids model download
    return proc


def test_publish_then_delete_fails_closed(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    _add_article(session, "keep", _unit(1), content="alpha")
    gone = _add_article(session, "gone", _unit(2), content="beta")

    result = proc.rebuild_index()
    assert result["published"], result
    assert result["indexed"] == 2

    # Search works while consistent.
    assert proc.search("alpha", relevance_threshold=0.0, max_results=5)

    # Delete an indexable article from a *separate* connection.
    raw = sqlite3.connect(engine.url.database)
    raw.execute("DELETE FROM articles WHERE id=?", (gone,))
    raw.commit()
    raw.close()

    # Fail closed: no results are served from the now-stale generation.
    assert proc.search("alpha", relevance_threshold=0.0, max_results=5) == []

    # Rebuild and only the survivor is indexed.
    rebuilt = proc.rebuild_index()
    assert rebuilt["published"]
    assert rebuilt["indexed"] == 1


def test_rowid_reuse_does_not_serve_stale_vector(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    aid = _add_article(session, "a", _unit(1), content="old content")
    assert proc.rebuild_index()["published"]

    raw = sqlite3.connect(engine.url.database)
    raw.execute("DELETE FROM articles WHERE id=?", (aid,))
    # Reuse the same rowid with a different vector.
    raw.execute(
        "INSERT INTO articles (id, feed_id, title, url, guid, content, processed, "
        "embedding_generated, embedding_vector) VALUES (?, 1, 'new', 'http://n', 'n', "
        "'new content', 1, 1, ?)",
        (aid, json.dumps(_unit(3))),
    )
    raw.commit()
    raw.close()

    assert proc.search("x", relevance_threshold=0.0, max_results=5) == []

    rebuilt = proc.rebuild_index()
    assert rebuilt["published"]
    # The new vector is served, not the stale one.
    results = proc.search("x", relevance_threshold=0.0, max_results=5)
    assert [r["id"] for r in results] == [aid]


def test_unconfirmed_publish_never_served(db, tmp_path, monkeypatch):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    _add_article(session, "a", _unit(1), content="content")
    assert proc.rebuild_index()["published"]
    assert proc.search("content", relevance_threshold=0.0)

    # Add a new eligible article (bumps generation), then force confirmation
    # to fail as if the generation changed mid-build.
    _add_article(session, "b", _unit(2), content="content b")

    manager = proc._get_index_manager()
    monkeypatch.setattr(manager, "_confirm", lambda **kwargs: False)
    result = manager.publish()
    assert result["published"] is False
    assert result["reason"] == "generation_changed"

    # The previous generation no longer matches the DB, so reads fail closed.
    assert proc.search("content", relevance_threshold=0.0) == []


def test_trigger_loss_detected_and_repaired(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    _add_article(session, "a", _unit(1), content="content")
    assert proc.rebuild_index()["published"]

    # Drop the triggers and mutate the table behind our back.
    raw = sqlite3.connect(engine.url.database)
    raw.execute("DROP TRIGGER vix_articles_au")
    raw.execute("UPDATE articles SET embedding_vector=? WHERE guid='a'", (json.dumps(_unit(4)),))
    raw.commit()
    raw.close()

    # Fingerprint no longer matches the published index -> fail closed.
    assert proc.search("content", relevance_threshold=0.0) == []

    # Repairing the schema reinstalls triggers and bumps the generation.
    ensure_schema(engine)
    state = VectorIndexManager(engine, config).read_state()
    assert state["current_generation"] > state["built_generation"]


def test_empty_publish_and_search(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    result = proc.rebuild_index()
    assert result["published"]
    assert result["indexed"] == 0
    assert result["n_items"] == 0

    state = VectorIndexManager(engine, config).read_state()
    assert state["status"] == "committed"
    assert state["annoy_n_items"] == 0
    assert proc.search("anything") == []


def test_flush_without_pending_is_noop(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    _add_article(session, "a", _unit(1), content="content")
    assert proc.rebuild_index()["published"]
    state_before = VectorIndexManager(engine, config).read_state()

    assert proc._flush_embeddings() == 0

    state_after = VectorIndexManager(engine, config).read_state()
    assert state_after["built_generation"] == state_before["built_generation"]
    assert state_after["status"] == "committed"
    assert proc.search("content", relevance_threshold=0.0)


def test_sparse_ids_validation(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    _add_article(session, "a", _unit(1), content="a")
    _add_article(session, "b", _unit(2), content="b")
    result = proc.rebuild_index()
    assert result["published"]
    # Annoy allocates max_id + 1 slots, not the vector count.
    assert result["n_items"] == 3
    assert result["indexed"] == 2
    assert proc.search("a", relevance_threshold=0.0)


def test_legacy_fixed_file_ignored(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    # Write a valid index at the legacy fixed path.
    from annoy import AnnoyIndex

    legacy = AnnoyIndex(DIM, "angular")
    legacy.add_item(1, _unit(1))
    legacy.build(4)
    legacy.save(config.annoy_index_path)

    _add_article(session, "a", _unit(2), content="content")
    proc = _processor(engine, config, session)

    # No committed generation yet: the legacy file must not be served.
    assert proc.search("content", relevance_threshold=0.0) == []


def test_long_lived_processor_sees_new_generation(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    _add_article(session, "a", _unit(1), content="first")
    assert proc.rebuild_index()["published"]
    assert len(proc.search("first", relevance_threshold=0.0)) == 1

    # Publish a new generation from another processor/session.
    other_session = sessionmaker(bind=engine)()
    other = _processor(engine, config, other_session)
    _add_article(other_session, "b", _unit(2), content="second")
    assert other.rebuild_index()["published"]
    other_session.close()

    # The original instance re-reads metadata and picks up the new generation.
    results = proc.search("second", relevance_threshold=0.0)
    assert len(results) >= 1


def test_trigger_semantics(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    manager = VectorIndexManager(engine, config)

    def gen():
        return manager.read_state()["current_generation"]

    # Un-embedded insert does not bump.
    start = gen()
    _add_article(session, "plain", generated=False)
    assert gen() == start

    # Embedded insert bumps.
    _add_article(session, "embedded", _unit(1))
    assert gen() == start + 1

    # No-op update does not bump.
    session.execute(text("UPDATE articles SET embedding_vector = embedding_vector"))
    session.commit()
    assert gen() == start + 1

    # Changing the vector bumps.
    session.execute(
        text("UPDATE articles SET embedding_vector = :v WHERE guid = 'embedded'"),
        {"v": json.dumps(_unit(2))},
    )
    session.commit()
    assert gen() == start + 2

    # Deleting an un-embedded article does not bump.
    session.execute(text("DELETE FROM articles WHERE guid = 'plain'"))
    session.commit()
    assert gen() == start + 2

    # Deleting an embedded article bumps.
    session.execute(text("DELETE FROM articles WHERE guid = 'embedded'"))
    session.commit()
    assert gen() == start + 3


def test_corrupt_file_fails_closed(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    _add_article(session, "a", _unit(1), content="content")
    assert proc.rebuild_index()["published"]

    manager = proc._get_index_manager()
    path = manager._file_path(manager.read_state())
    with open(path, "r+b") as fh:
        fh.truncate(8)

    proc._invalidate_vector_index_cache()
    assert proc.search("content", relevance_threshold=0.0) == []


def test_cli_rebuild_and_refresh_revalidate(db, tmp_path, monkeypatch):
    """search(refresh=True) revalidates without rebuilding; rebuild_index publishes."""
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    _add_article(session, "a", _unit(1), content="content")
    assert proc.search("content", relevance_threshold=0.0) == []

    gen_before = VectorIndexManager(engine, config).read_state()["current_generation"]
    # refresh must not publish anything on its own.
    assert proc.search("content", relevance_threshold=0.0, refresh=True) == []
    assert VectorIndexManager(engine, config).read_state()["current_generation"] == gen_before

    assert proc.rebuild_index()["published"]
    assert proc.search("content", relevance_threshold=0.0, refresh=True)


def test_ensure_index_current_repairs_stale(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    proc = _processor(engine, config, session)

    _add_article(session, "a", _unit(1), content="alpha")
    _add_article(session, "b", _unit(2), content="beta")
    assert proc.rebuild_index()["published"]

    # External deletion makes the index stale.
    raw = sqlite3.connect(engine.url.database)
    raw.execute("DELETE FROM articles WHERE guid='b'")
    raw.commit()
    raw.close()
    assert proc.search("alpha", relevance_threshold=0.0) == []

    result = proc.ensure_index_current()
    assert result["rebuilt"]
    assert result["indexed"] == 1
    assert len(proc.search("alpha", relevance_threshold=0.0)) == 1

    # Already current: no rebuild.
    again = proc.ensure_index_current()
    assert again["rebuilt"] is False


def test_concurrent_publish_same_generation_loses_safely(db, tmp_path):
    engine, session, feed_id = db
    config = FakeConfig(tmp_path)
    _add_article(session, "a", _unit(1), content="content")

    manager = VectorIndexManager(engine, config)
    first = manager.publish()
    second = manager.publish()
    assert first["published"] and second["published"]

    # Both confirmed at the same generation; the last-confirmed file is served.
    state = manager.read_state()
    assert state["index_filename"] == second["filename"]
    index, reason, _ = manager.load_validated()
    assert reason is None
    assert index.get_n_items() == state["annoy_n_items"]
