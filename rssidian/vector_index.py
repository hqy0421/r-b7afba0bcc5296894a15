"""
Generation-aware management of the Annoy vector index.

SQLite is the source of truth.  Every published index file is tied to a
``generation`` that is maintained by database triggers and recorded in the
single-row ``vector_index_meta`` table.  Readers only load an index whose
recorded generation still matches the database, so index entries that refer to
deleted (or otherwise rewritten) article rows are never served.

The module deliberately does not import ``models`` or ``core`` to avoid import
cycles; it operates on a SQLAlchemy ``Engine`` and a config object.
"""

import glob
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# A row participates in the vector index only when both flags/values are set.
INDEXABLE_PREDICATE = "embedding_generated = 1 AND embedding_vector IS NOT NULL"

META_TABLE_NAME = "vector_index_meta"

META_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {META_TABLE_NAME} (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    db_uuid TEXT,
    current_generation INTEGER NOT NULL DEFAULT 0,
    built_generation INTEGER,
    index_filename TEXT,
    status TEXT NOT NULL DEFAULT 'empty',
    embedding_dim INTEGER,
    embedding_model TEXT,
    metric TEXT,
    annoy_n_items INTEGER,
    indexed_count INTEGER,
    eligible_count INTEGER,
    skipped_count INTEGER,
    fingerprint TEXT,
    built_at TEXT
)
"""

# Only transitions into or out of the indexable set (and vector/id edits of an
# indexable row) bump the generation.  Deleting an un-embedded article must not
# invalidate a perfectly good index.
TRIGGER_DDLS = {
    "vix_articles_ai": (
        "CREATE TRIGGER vix_articles_ai AFTER INSERT ON articles "
        "WHEN NEW.embedding_generated = 1 AND NEW.embedding_vector IS NOT NULL "
        "BEGIN UPDATE vector_index_meta SET current_generation = current_generation + 1 "
        "WHERE id = 1; END"
    ),
    "vix_articles_au": (
        "CREATE TRIGGER vix_articles_au AFTER UPDATE OF id, embedding_vector, embedding_generated "
        "ON articles "
        "WHEN ((NEW.embedding_generated = 1 AND NEW.embedding_vector IS NOT NULL) "
        "      IS NOT (OLD.embedding_generated = 1 AND OLD.embedding_vector IS NOT NULL)) "
        "  OR NEW.embedding_vector IS NOT OLD.embedding_vector "
        "  OR NEW.id IS NOT OLD.id "
        "BEGIN UPDATE vector_index_meta SET current_generation = current_generation + 1 "
        "WHERE id = 1; END"
    ),
    "vix_articles_ad": (
        "CREATE TRIGGER vix_articles_ad AFTER DELETE ON articles "
        "WHEN OLD.embedding_generated = 1 AND OLD.embedding_vector IS NOT NULL "
        "BEGIN UPDATE vector_index_meta SET current_generation = current_generation + 1 "
        "WHERE id = 1; END"
    ),
}

STATE_SQL = f"""
SELECT m.*,
       (SELECT COUNT(*) FROM articles WHERE {INDEXABLE_PREDICATE}) AS current_count,
       (SELECT COALESCE(SUM(id), 0) FROM articles WHERE {INDEXABLE_PREDICATE}) AS current_id_sum,
       (SELECT COALESCE(MIN(id), 0) FROM articles WHERE {INDEXABLE_PREDICATE}) AS current_min_id,
       (SELECT COALESCE(MAX(id), 0) FROM articles WHERE {INDEXABLE_PREDICATE}) AS current_max_id,
       (SELECT COUNT(*) FROM sqlite_master
          WHERE type = 'trigger'
            AND name IN ('{"', '".join(name for name in TRIGGER_DDLS)}')) AS trigger_count
FROM {META_TABLE_NAME} m
WHERE m.id = 1
"""

ROWS_SQL = f"SELECT id, embedding_vector FROM articles WHERE {INDEXABLE_PREDICATE} ORDER BY id"

CONFIRM_SQL = f"""
UPDATE {META_TABLE_NAME}
SET built_generation = :generation,
    status = 'committed',
    index_filename = :filename,
    embedding_dim = :dim,
    embedding_model = :model,
    metric = :metric,
    annoy_n_items = :n_items,
    indexed_count = :indexed,
    eligible_count = :eligible,
    skipped_count = :skipped,
    fingerprint = :fingerprint,
    built_at = :built_at
WHERE id = 1
  AND db_uuid = :db_uuid
  AND current_generation = :generation
  AND (SELECT COUNT(*) FROM articles WHERE {INDEXABLE_PREDICATE}) = :eligible
  AND (SELECT COALESCE(SUM(id), 0) FROM articles WHERE {INDEXABLE_PREDICATE}) = :id_sum
  AND (SELECT COALESCE(MIN(id), 0) FROM articles WHERE {INDEXABLE_PREDICATE}) = :min_id
  AND (SELECT COALESCE(MAX(id), 0) FROM articles WHERE {INDEXABLE_PREDICATE}) = :max_id
"""


def fingerprint(count: int, id_sum: int, min_id: int, max_id: int) -> str:
    """Cheap aggregate fingerprint over the indexable article set."""
    return f"{count}:{id_sum}:{min_id}:{max_id}"


def _normalize_sql(sql: Optional[str]) -> str:
    normalized = " ".join((sql or "").lower().split())
    # SQLite reports the stored trigger body without the trailing semicolon.
    return normalized.rstrip(";").strip()


def ensure_schema(engine: Engine) -> None:
    """
    Create (or repair) the metadata table and generation triggers.

    Gated by ``PRAGMA user_version`` and trigger inspection so the common case
    (schema already correct) performs only reads.  If triggers were removed or
    altered behind our back (table rebuild, dump/restore, sqlite-utils), they are
    reinstalled and the generation is bumped so any existing index fails closed.
    """
    with engine.begin() as conn:
        version = conn.exec_driver_sql("PRAGMA user_version").scalar() or 0
        has_table = (
            conn.exec_driver_sql(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name",
                {"name": META_TABLE_NAME},
            ).first()
            is not None
        )
        existing_triggers = {
            row[0]: row[1]
            for row in conn.exec_driver_sql(
                "SELECT name, sql FROM sqlite_master WHERE type='trigger'"
            ).all()
        }
        triggers_ok = all(
            _normalize_sql(existing_triggers.get(name)) == _normalize_sql(ddl)
            for name, ddl in TRIGGER_DDLS.items()
        )
        has_row = has_table and (
            conn.exec_driver_sql(
                "SELECT 1 FROM vector_index_meta WHERE id = 1"
            ).first()
            is not None
        )

        if has_table and has_row and version >= SCHEMA_VERSION and triggers_ok:
            return

        repaired = has_table and (not triggers_ok or not has_row)
        conn.exec_driver_sql(META_TABLE_DDL)
        conn.exec_driver_sql(
            "INSERT OR IGNORE INTO vector_index_meta "
            "(id, db_uuid, current_generation, status) VALUES (1, ?, 0, 'empty')",
            (uuid.uuid4().hex,),
        )

        for name in TRIGGER_DDLS:
            conn.exec_driver_sql(f"DROP TRIGGER IF EXISTS {name}")
        for ddl in TRIGGER_DDLS.values():
            conn.exec_driver_sql(ddl)

        if repaired:
            conn.exec_driver_sql(
                "UPDATE vector_index_meta SET current_generation = current_generation + 1 "
                "WHERE id = 1"
            )
            logger.warning(
                "Vector index generation triggers were missing or changed; "
                "generation bumped so stale indexes are not served."
            )

        conn.exec_driver_sql(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")


class VectorIndexManager:
    """Publishes and validates generation-addressed Annoy index files."""

    def __init__(self, engine: Engine, config: Any):
        self.engine = engine
        self.config = config

    # ------------------------------------------------------------------ paths
    def _base_path(self) -> str:
        return os.path.abspath(os.path.expanduser(self.config.annoy_index_path))

    def _split_path(self) -> Tuple[str, str]:
        base = self._base_path()
        directory = os.path.dirname(base) or "."
        stem = os.path.splitext(os.path.basename(base))[0] or "annoy"
        return directory, stem

    def _filename(self, db_uuid: str, generation: int, nonce: str) -> str:
        _, stem = self._split_path()
        return f"{stem}.{db_uuid}.g{generation}.{nonce}.idx"

    @staticmethod
    def _is_safe_filename(filename: Optional[str]) -> bool:
        if not filename or not filename.endswith(".idx"):
            return False
        return os.path.basename(filename) == filename

    # ------------------------------------------------------------- meta reads
    def read_state(self) -> Optional[Dict[str, Any]]:
        """Read meta plus current DB fingerprint on a fresh connection."""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text(STATE_SQL)).mappings().first()
        except Exception as exc:  # table missing / legacy database
            logger.debug("Unable to read vector index metadata: %s", exc)
            return None
        if not row:
            return None
        state = dict(row)
        state["current_fingerprint"] = fingerprint(
            state["current_count"],
            state["current_id_sum"],
            state["current_min_id"],
            state["current_max_id"],
        )
        return state

    def _config_matches(self, state: Dict[str, Any]) -> bool:
        return (
            state.get("embedding_dim") == self.config.embedding_dim
            and state.get("embedding_model") == self.config.embedding_model_name
            and state.get("metric") == self.config.annoy_metric
        )

    def _file_path(self, state: Dict[str, Any]) -> Optional[str]:
        filename = state.get("index_filename")
        if not self._is_safe_filename(filename):
            return None
        directory, _ = self._split_path()
        return os.path.join(directory, filename)

    def is_current(self, state: Optional[Dict[str, Any]]) -> bool:
        """Whether a committed index is consistent with the current database."""
        if not state:
            return False
        if state.get("status") != "committed":
            return False
        built = state.get("built_generation")
        if built is None or built != state.get("current_generation"):
            return False
        # Triggers missing/altered means the generation can no longer be trusted.
        if state.get("trigger_count") != len(TRIGGER_DDLS):
            return False
        if not self._config_matches(state):
            return False
        if state.get("fingerprint") != state.get("current_fingerprint"):
            return False
        path = self._file_path(state)
        return bool(path and os.path.exists(path))

    def load_validated(
        self, expected_uuid: Optional[str] = None
    ) -> Tuple[Optional[Any], Optional[str], Optional[Dict[str, Any]]]:
        """
        Load the committed index only if it matches the database.

        Returns ``(index, reason, state)``; ``index`` is ``None`` (fail closed)
        whenever anything is inconsistent.  Never falls back to a previous or
        unconfirmed file.
        """
        state = self.read_state()
        if state is None:
            return None, "schema_missing", None
        if expected_uuid and state.get("db_uuid") != expected_uuid:
            return None, "database_changed", state
        if state.get("status") != "committed":
            return None, "uninitialized", state
        if state.get("built_generation") != state.get("current_generation"):
            return None, "stale", state
        if state.get("trigger_count") != len(TRIGGER_DDLS):
            return None, "triggers_missing", state
        if not self._config_matches(state):
            return None, "config_mismatch", state
        if state.get("fingerprint") != state.get("current_fingerprint"):
            return None, "fingerprint_mismatch", state
        path = self._file_path(state)
        if not path or not os.path.exists(path):
            return None, "file_missing", state

        try:
            index = self._new_index()
            index.load(path)
            if index.get_n_items() != state.get("annoy_n_items"):
                return None, "size_mismatch", state
        except Exception as exc:
            logger.warning("Failed to load vector index %s: %s", path, exc)
            return None, "load_failed", state
        return index, None, state

    def _new_index(self):
        from annoy import AnnoyIndex

        return AnnoyIndex(self.config.embedding_dim, self.config.annoy_metric)

    # ---------------------------------------------------------------- publish
    def publish(self) -> Dict[str, Any]:
        """
        Build an index from a single database snapshot and publish it.

        The index file is written to a unique, generation-addressed name; the
        metadata row is then confirmed with a conditional UPDATE that also
        re-verifies the snapshot's generation and fingerprint.  If the database
        changed during the build, or another writer got there first, the file is
        removed and never referenced.
        """
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text(STATE_SQL)).mappings().first()
                rows = conn.execute(text(ROWS_SQL)).all()
        except Exception as exc:
            logger.error("Unable to snapshot database for vector index: %s", exc)
            return {"published": False, "reason": "schema_missing", "indexed": 0}

        if not row:
            logger.error("Vector index metadata row is missing")
            return {"published": False, "reason": "schema_missing", "indexed": 0}

        state = dict(row)
        generation = state["current_generation"]
        db_uuid = state["db_uuid"]
        eligible = state["current_count"]
        id_sum = state["current_id_sum"]
        min_id = state["current_min_id"]
        max_id = state["current_max_id"]
        fp = fingerprint(eligible, id_sum, min_id, max_id)

        dim = self.config.embedding_dim
        index = self._new_index()
        indexed = 0
        skipped = 0
        for article_id, vector_json in rows:
            try:
                vector = json.loads(vector_json)
                if not isinstance(vector, list) or len(vector) != dim:
                    raise ValueError("unexpected embedding dimension")
                index.add_item(article_id, vector)
                indexed += 1
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipping article %s in index build: %s", article_id, exc)
                skipped += 1

        index.build(self.config.annoy_n_trees)
        n_items = index.get_n_items()

        directory, _ = self._split_path()
        os.makedirs(directory, exist_ok=True)
        filename = self._filename(db_uuid, generation, uuid.uuid4().hex[:12])
        final_path = os.path.join(directory, filename)

        try:
            self._save_atomically(index, final_path, directory)
        except Exception as exc:
            logger.error("Failed to write vector index %s: %s", final_path, exc)
            return {
                "published": False,
                "reason": "write_failed",
                "indexed": indexed,
                "generation": generation,
            }

        confirmed = self._confirm(
            generation=generation,
            db_uuid=db_uuid,
            eligible=eligible,
            id_sum=id_sum,
            min_id=min_id,
            max_id=max_id,
            filename=filename,
            n_items=n_items,
            indexed=indexed,
            skipped=skipped,
            fp=fp,
        )

        if not confirmed:
            try:
                os.remove(final_path)
            except OSError:
                pass
            logger.warning(
                "Vector index generation changed during build; discarded unconfirmed file %s",
                filename,
            )
            return {
                "published": False,
                "reason": "generation_changed",
                "indexed": indexed,
                "generation": generation,
            }

        self._cleanup_old_files(db_uuid, generation, keep_filename=filename)
        logger.info(
            "Published vector index generation %s (%s vectors, %s items)",
            generation,
            indexed,
            n_items,
        )
        return {
            "published": True,
            "reason": None,
            "indexed": indexed,
            "skipped": skipped,
            "eligible": eligible,
            "n_items": n_items,
            "generation": generation,
            "filename": filename,
        }

    def _save_atomically(self, index: Any, final_path: str, directory: str) -> None:
        tmp_path = f"{final_path}.tmp{os.getpid()}"
        try:
            index.save(tmp_path)
            fd = os.open(tmp_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp_path, final_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    def _confirm(
        self,
        generation: int,
        db_uuid: str,
        eligible: int,
        id_sum: int,
        min_id: int,
        max_id: int,
        filename: str,
        n_items: int,
        indexed: int,
        skipped: int,
        fp: str,
    ) -> bool:
        params = {
            "generation": generation,
            "filename": filename,
            "dim": self.config.embedding_dim,
            "model": self.config.embedding_model_name,
            "metric": self.config.annoy_metric,
            "n_items": n_items,
            "indexed": indexed,
            "eligible": eligible,
            "skipped": skipped,
            "fingerprint": fp,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "db_uuid": db_uuid,            "id_sum": id_sum,
            "min_id": min_id,
            "max_id": max_id,
        }
        with self.engine.begin() as conn:
            result = conn.execute(text(CONFIRM_SQL), params)
            return result.rowcount == 1

    def _cleanup_old_files(
        self, db_uuid: str, committed_generation: int, keep_filename: Optional[str] = None
    ) -> None:
        directory, stem = self._split_path()
        prefix = f"{stem}.{db_uuid}.g"
        now = time.time()
        for path in glob.glob(os.path.join(directory, f"{prefix}*.idx")):
            name = os.path.basename(path)
            try:
                tail = name[len(prefix):]
                gen = int(tail.split(".", 1)[0])
            except (ValueError, IndexError):
                continue
            if gen < committed_generation:
                self._remove(path)
            elif gen == committed_generation and name != keep_filename:
                # Same-generation orphan from a concurrent builder (both were
                # valid snapshots; only one got confirmed).  Only delete once it
                # is old enough that no reader can still be opening it.
                try:
                    if now - os.path.getmtime(path) > 300:
                        self._remove(path)
                except OSError:
                    pass

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass
