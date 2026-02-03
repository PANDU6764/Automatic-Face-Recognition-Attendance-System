from pathlib import Path
import sqlite3
import json
from datetime import datetime, date

def connect(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        person_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        name TEXT NOT NULL,
        ts TEXT NOT NULL,
        day TEXT NOT NULL,
        UNIQUE(person_id, day)
    );
    """)
    conn.commit()

def upsert_person(conn, person_id: str, name: str, embedding):
    emb_json = json.dumps([float(x) for x in embedding])
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute("""
    INSERT INTO persons(person_id, name, embedding_json, created_at)
    VALUES(?, ?, ?, ?)
    ON CONFLICT(person_id) DO UPDATE SET
        name=excluded.name,
        embedding_json=excluded.embedding_json;
    """, (person_id, name, emb_json, now))
    conn.commit()

def load_persons(conn):
    cur = conn.execute("SELECT person_id, name, embedding_json FROM persons;")
    rows = cur.fetchall()
    persons = []
    for pid, name, emb_json in rows:
        persons.append((pid, name, json.loads(emb_json)))
    return persons

def mark_attendance(conn, person_id: str, name: str):
    now = datetime.now().isoformat(timespec="seconds")
    day = date.today().isoformat()
    conn.execute("""
    INSERT OR IGNORE INTO attendance(person_id, name, ts, day)
    VALUES(?, ?, ?, ?);
    """, (person_id, name, now, day))
    conn.commit()
