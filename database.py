import sqlite3
from pathlib import Path
from typing import Any

from flask import g


DB_PATH: Path | None = None


def init_db(path: Path) -> None:
    global DB_PATH
    DB_PATH = path
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                prompt TEXT NOT NULL,
                genre TEXT NOT NULL,
                mood TEXT NOT NULL,
                instrument TEXT NOT NULL,
                duration INTEGER NOT NULL,
                midi_file TEXT NOT NULL,
                wav_file TEXT,
                mp3_file TEXT,
                favorite INTEGER NOT NULL DEFAULT 0,
                listening_time INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_songs_created_at ON songs(created_at);
            CREATE INDEX IF NOT EXISTS idx_songs_favorite ON songs(favorite);
            """
        )
        _migrate_songs_table(conn)


def _migrate_songs_table(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(songs)").fetchall()}
    columns = {
        "wav_file": "TEXT",
        "mp3_file": "TEXT",
        "favorite": "INTEGER NOT NULL DEFAULT 0",
        "listening_time": "INTEGER NOT NULL DEFAULT 0",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE songs ADD COLUMN {name} {definition}")


def get_db() -> sqlite3.Connection:
    if DB_PATH is None:
        raise RuntimeError("Database is not initialized.")
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_exception: BaseException | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def insert_song(
    *,
    prompt: str,
    genre: str,
    mood: str,
    instrument: str,
    duration: int,
    title: str,
    midi_file: str,
    wav_file: str | None,
    mp3_file: str | None,
) -> int:
    conn = get_db()
    cursor = conn.execute(
        """
        INSERT INTO songs (title, prompt, genre, mood, instrument, duration, midi_file, wav_file, mp3_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (title, prompt, genre, mood, instrument, duration, midi_file, wav_file, mp3_file),
    )
    conn.commit()
    return int(cursor.lastrowid)


def list_songs(search: str = "", favorites: bool = False) -> list[dict[str, Any]]:
    query = "SELECT * FROM songs WHERE 1 = 1"
    params: list[Any] = []
    if search:
        query += " AND (title LIKE ? OR prompt LIKE ? OR genre LIKE ? OR mood LIKE ?)"
        needle = f"%{search}%"
        params.extend([needle, needle, needle, needle])
    if favorites:
        query += " AND favorite = 1"
    query += " ORDER BY created_at DESC, id DESC"
    return [row_to_dict(row) for row in get_db().execute(query, params).fetchall()]


def toggle_favorite(song_id: int) -> bool:
    conn = get_db()
    row = conn.execute("SELECT favorite FROM songs WHERE id = ?", (song_id,)).fetchone()
    if not row:
        raise ValueError("Song not found.")
    favorite = 0 if row["favorite"] else 1
    conn.execute("UPDATE songs SET favorite = ? WHERE id = ?", (favorite, song_id))
    conn.commit()
    return bool(favorite)


def rename_song(song_id: int, title: str) -> None:
    conn = get_db()
    conn.execute("UPDATE songs SET title = ? WHERE id = ?", (title, song_id))
    conn.commit()


def delete_song(song_id: int) -> None:
    conn = get_db()
    conn.execute("DELETE FROM songs WHERE id = ?", (song_id,))
    conn.commit()


def update_listening_time(song_id: int, seconds: int) -> None:
    if song_id <= 0 or seconds <= 0:
        return
    conn = get_db()
    conn.execute("UPDATE songs SET listening_time = listening_time + ? WHERE id = ?", (seconds, song_id))
    conn.commit()


def get_analytics() -> dict[str, Any]:
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) AS count FROM songs").fetchone()["count"]
    listen = conn.execute("SELECT COALESCE(SUM(listening_time), 0) AS total FROM songs").fetchone()["total"]
    top_genre = conn.execute(
        "SELECT genre, COUNT(*) AS count FROM songs GROUP BY genre ORDER BY count DESC LIMIT 1"
    ).fetchone()
    top_mood = conn.execute(
        "SELECT mood, COUNT(*) AS count FROM songs GROUP BY mood ORDER BY count DESC LIMIT 1"
    ).fetchone()
    genre_rows = conn.execute("SELECT genre AS label, COUNT(*) AS value FROM songs GROUP BY genre").fetchall()
    mood_rows = conn.execute("SELECT mood AS label, COUNT(*) AS value FROM songs GROUP BY mood").fetchall()
    recent_rows = conn.execute(
        "SELECT DATE(created_at) AS label, COUNT(*) AS value FROM songs GROUP BY DATE(created_at) ORDER BY label"
    ).fetchall()
    return {
        "songs_generated": total,
        "most_used_genre": top_genre["genre"] if top_genre else "None",
        "most_used_mood": top_mood["mood"] if top_mood else "None",
        "total_listening_time": listen,
        "genres": [row_to_dict(row) for row in genre_rows],
        "moods": [row_to_dict(row) for row in mood_rows],
        "recent": [row_to_dict(row) for row in recent_rows],
    }
