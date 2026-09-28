"""
Persistence layer for multimodal learner assistant.

Stores learner profiles, interaction history, feedback, indexed-document
metadata, and pending clarification state. Existing databases are migrated
in place when new columns are introduced.
"""
import json
import sqlite3

DB_PATH = 'learner.db'

"""
Maximum age of an unanswered clarification request before it is discarded.
This prevents an old clarification from being applied to a later, unrelated
user message.
"""
PENDING_CLARIFICATION_MAX_AGE_SECONDS = 15 * 60


def _normalize_user_id(user_id):
    """
    Normalizes user IDs before database access so equivalent identifiers map to
    the same profile and interaction history.
    An example would be "alice", "Alice", and "alice " (with trailing whitespace)
    """
    return (user_id or "").strip().lower()


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Creates the database tables required by the learner memory system and
    applies guarded schema migrations for existing databases.
    """
    conn = _connect()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_profile
                 (user_id TEXT PRIMARY KEY,
                  learning_style TEXT,
                  difficulty TEXT,
                  topics TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS interactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  query TEXT,
                  answer TEXT,
                  topic TEXT,
                  feedback INTEGER)''')
    
    """
    Registry of files successfully indexed for each user.

    The content hash detects identical files even when filenames differ. The
    filename is also stored so that reusing an existing filename can be detected
    even when the file contents have changed.
    """
    c.execute('''CREATE TABLE IF NOT EXISTS indexed_files
                 (user_id TEXT,
                  filename TEXT,
                  sha256 TEXT,
                  chunks INTEGER,
                  indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY (user_id, sha256))''')

    """
    Stores the unresolved clarification associated with each user. The original
    query is retained so that a follow-up selection can be resolved against the
    question that triggered the clarification. Only one unresolved clarification
    is stored per user.
    """
    c.execute('''CREATE TABLE IF NOT EXISTS pending_clarifications
                 (user_id TEXT PRIMARY KEY,
                  query TEXT,
                  candidates TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  attempts INTEGER DEFAULT 0)''')
    
    """
    Counts unresolved replies to a pending clarification. This  
    allows the engine to limit repeated clarification prompts.
    """
    c.execute("PRAGMA table_info(pending_clarifications)")
    if "attempts" not in {row[1] for row in c.fetchall()}:
        c.execute("ALTER TABLE pending_clarifications ADD COLUMN attempts INTEGER DEFAULT 0")

    # Add the column for databases created before attempts was introduced.
    c.execute("PRAGMA table_info(interactions)")
    existing_columns = {row[1] for row in c.fetchall()}
    
    """
    `rateable = 0` identifies interactions that should not contribute to learner
    feedback, such as clarifications, system errors, and responses stating that
    the requested information is not present in the learning materials.

    These interactions are excluded so that system failures are not interpreted
    as evidence of weak learner understanding.
    """
    if "rateable" not in existing_columns:
        c.execute("ALTER TABLE interactions ADD COLUMN rateable INTEGER DEFAULT 1")
        
    """
    Source document each answer was retrieved from, so weak topics can be
    aggregated per lecture rather than per extracted keyword. NULL for rows
    logged before this column existed (they are excluded from aggregation).
    """
    if "source" not in existing_columns:
        c.execute("ALTER TABLE interactions ADD COLUMN source TEXT")
    conn.commit()
    conn.close()


def get_profile(user_id="demo"):
    """
    Returns the stored learning profile for a user, or None if no profile exists.
    """ 
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM user_profile WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"learning_style": row[1], "difficulty": row[2], "topics": row[3].split(",")}
    return None


def save_profile(user_id, learning_style, difficulty, topics):
    """
    Creates or replaces the stored learning profile for a user.
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO user_profile VALUES (?,?,?,?)",
              (user_id, learning_style, difficulty, ",".join(topics)))
    conn.commit()
    conn.close()


def log_interaction(user_id, query, answer, topic, feedback=None, rateable=1, source=None):
    """
    Stores one learner interaction and its optional feedback metadata.
    
    `rateable` defaults to 1 for backward compatibility with existing callers.
    Non-answer interactions should explicitly pass `rateable=0`.
    """
    
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("INSERT INTO interactions (user_id, query, answer, topic, feedback, rateable, source) "
              "VALUES (?,?,?,?,?,?,?)",
              (user_id, query, answer, topic, feedback, 1 if rateable else 0, source))
    conn.commit()
    conn.close()


# Personalisation engine
def get_weak_topics(user_id, threshold=3):
    """
    Returns source documents associated with at least `threshold` low-rated
    interactions.

    Interactions are grouped by source document rather than extracted keyword
    because source-level aggregation provides sufficient observations for the
    current personalization strategy. A document is flagged based on the count of
    low ratings, while the average rating is retained for recommendation context.

    Interactions without a source are excluded.
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT source,
               SUM(CASE WHEN feedback <= 2 THEN 1 ELSE 0 END) AS n_low,
               AVG(feedback) AS avg_fb
        FROM interactions
        WHERE user_id = ? AND feedback IS NOT NULL
              AND source IS NOT NULL AND source <> ''
        GROUP BY source
        HAVING n_low >= ?
        ORDER BY n_low DESC, avg_fb ASC
    """, (user_id, threshold))
    rows = c.fetchall()
    conn.close()
    return rows

# Feedback persistence ------------------------------------------------------------------------
def get_last_interaction_id(user_id):
    """ 
    Returns the ID of the user's most recent interaction, or None if no interactions exist.
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id FROM interactions WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_last_rateable_interaction_id(user_id):
    """
    Returns the most recent rateable interaction, excluding system responses such
    as clarifications and non-answer messages.

    This is required because feedback is submitted after the interaction is
    logged, so the latest interaction is not necessarily the latest answer that
    can legitimately receive feedback.
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id FROM interactions WHERE user_id=? AND rateable=1 "
              "ORDER BY id DESC LIMIT 1", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_last_interaction(user_id, rateable_only=False):
    """
    Returns the most recent interaction as a dictionary containing its ID, query,
    answer, topic, and feedback, or None if no interaction exists.

    When `rateable_only=True`, non-rateable interactions are excluded.
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    sql = ("SELECT id, query, answer, topic, feedback FROM interactions "
           "WHERE user_id=?")
    if rateable_only:
        sql += " AND rateable=1"
    sql += " ORDER BY id DESC LIMIT 1"
    c.execute(sql, (user_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "query": row[1], "answer": row[2],
            "topic": row[3], "feedback": row[4]}


def set_feedback(interaction_id, feedback):
    """
    Records a user's rating (1-5) against a specific interaction row.
    """
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE interactions SET feedback=? WHERE id=?", (feedback, interaction_id))
    conn.commit()
    conn.close()


# Indexed-file registry ------------------------------------------------------------------------

def is_file_indexed(user_id, sha256=None, filename=None):
    """
    Returns an indexed-file record if the user has already registered the same
    content hash or filename.

    The content hash detects identical content under different filenames. The
    filename check also prevents a changed file from reusing an existing source
    name, which could make source-level retrieval ambiguous.
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT filename, sha256, chunks, indexed_at FROM indexed_files "
              "WHERE user_id=? AND (sha256=? OR filename=?)",
              (user_id, sha256 or "", filename or ""))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"filename": row[0], "sha256": row[1], "chunks": row[2], "indexed_at": row[3]}


def record_indexed_file(user_id, filename, sha256, chunks):
    # Records a file after successful indexing so failed parsing or
    # indexing does not prevent the file from being retried.
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO indexed_files (user_id, filename, sha256, chunks) "
              "VALUES (?,?,?,?)", (user_id, filename, sha256, chunks))
    conn.commit()
    conn.close()


def list_indexed_files(user_id):
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT filename, chunks, indexed_at FROM indexed_files "
              "WHERE user_id=? ORDER BY indexed_at", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [{"filename": r[0], "chunks": r[1], "indexed_at": r[2]} for r in rows]


# Pending clarification ------------------------------------------------------------------------

def set_pending_clarification(user_id, query, candidates):
    """
    Stores the question being clarified and and its candidate documents.
    Replaces any previous unanswered clarification for this user.
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO pending_clarifications (user_id, query, candidates) "
              "VALUES (?,?,?)", (user_id, query, json.dumps(list(candidates))))
    conn.commit()
    conn.close()


def get_pending_clarification(user_id, max_age_seconds=PENDING_CLARIFICATION_MAX_AGE_SECONDS):
    """
    Returns the user's pending clarification, or None if none exists or it has
    expired.

    Expired clarifications are removed from the database before returning.
    
    Expiration is calculated using SQLite's current UTC time so the age comparison
    does not depend on the application's local timezone.
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT query, candidates, (julianday('now') - julianday(created_at)) * 86400.0, "
              "COALESCE(attempts, 0) FROM pending_clarifications WHERE user_id=?", (user_id,))
    row = c.fetchone()
    if row and row[2] is not None and row[2] > max_age_seconds:
        c.execute("DELETE FROM pending_clarifications WHERE user_id=?", (user_id,))
        conn.commit()
        row = None
    conn.close()
    if not row:
        return None
    try:
        candidates = json.loads(row[1])
    except (TypeError, ValueError):
        return None
    return {"query": row[0], "candidates": candidates, "attempts": row[3]}


def bump_pending_attempts(user_id):
    """
    Increments the unresolved-attempt counter for a pending clarification 
    and returns the new count. Returns 0 if no clarification is pending.
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE pending_clarifications SET attempts = COALESCE(attempts, 0) + 1 "
              "WHERE user_id=?", (user_id,))
    c.execute("SELECT attempts FROM pending_clarifications WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.commit()
    conn.close()
    return row[0] if row else 0


def clear_pending_clarification(user_id):
    """ 
    Removes the user's pending clarification, if one exists
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("DELETE FROM pending_clarifications WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def clear_indexed_files(user_id):
    """
    Removes all indexed-file registry entries for the user.

    The caller must also clear the corresponding vector-store collection so that
    the registry and vector store remain consistent.
    """
    user_id = _normalize_user_id(user_id)
    conn = _connect()
    c = conn.cursor()
    c.execute("DELETE FROM indexed_files WHERE user_id=?", (user_id,))
    deleted = c.rowcount
    # a pending "which document?" question refers to documents that are now gone
    c.execute("DELETE FROM pending_clarifications WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
    return deleted