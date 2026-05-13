import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "portfolio.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            type TEXT,
            title TEXT,
            raw_content TEXT,
            summary TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    # Migrate existing DBs that predate the summary column
    existing = [r[1] for r in c.execute("PRAGMA table_info(sources)").fetchall()]
    if "summary" not in existing:
        c.execute("ALTER TABLE sources ADD COLUMN summary TEXT")

    c.execute("""
        CREATE TABLE IF NOT EXISTS extracted_companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
            ticker TEXT,
            company_name TEXT,
            sub_theme TEXT,
            sentiment TEXT,
            context TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE,
            company_name TEXT,
            sub_themes TEXT,
            source_type TEXT,
            notes TEXT,
            added_at TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS wma_cache (
            ticker TEXT PRIMARY KEY,
            company_name TEXT,
            index_name TEXT,
            current_price REAL,
            wma_200 REAL,
            pct_from_wma REAL,
            last_updated TEXT
        )
    """)

    conn.commit()
    conn.close()


# ── Sources ──────────────────────────────────────────────────────────────────

def save_source(url, type_, title, raw_content, summary=None):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO sources (url, type, title, raw_content, summary) VALUES (?, ?, ?, ?, ?)",
        (url, type_, title, raw_content, summary)
    )
    source_id = c.lastrowid
    conn.commit()
    conn.close()
    return source_id


def get_all_sources():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sources ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_source(source_id):
    conn = get_connection()
    conn.execute("DELETE FROM extracted_companies WHERE source_id = ?", (source_id,))
    conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
    conn.commit()
    conn.close()


# ── Extracted companies ───────────────────────────────────────────────────────

def save_extracted_companies(source_id, companies):
    conn = get_connection()
    for c in companies:
        conn.execute(
            """INSERT INTO extracted_companies
               (source_id, ticker, company_name, sub_theme, sentiment, context)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                source_id,
                c.get("ticker", "UNKNOWN"),
                c.get("company_name", ""),
                c.get("sub_theme", "unknown"),
                c.get("sentiment", "neutral"),
                c.get("context", ""),
            )
        )
    conn.commit()
    conn.close()


def get_all_extracted_companies():
    conn = get_connection()
    rows = conn.execute("""
        SELECT ec.*, s.title as source_title, s.type as source_type, s.url as source_url
        FROM extracted_companies ec
        JOIN sources s ON ec.source_id = s.id
        ORDER BY ec.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_companies_by_source(source_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM extracted_companies WHERE source_id = ?", (source_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Watchlist ─────────────────────────────────────────────────────────────────

def add_to_watchlist(ticker, company_name, sub_themes, source_type, notes=""):
    conn = get_connection()
    themes_json = json.dumps(sub_themes) if isinstance(sub_themes, list) else sub_themes
    try:
        conn.execute(
            """INSERT INTO watchlist (ticker, company_name, sub_themes, source_type, notes)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
                 company_name=excluded.company_name,
                 sub_themes=excluded.sub_themes,
                 source_type=CASE
                   WHEN watchlist.source_type != excluded.source_type THEN 'both'
                   ELSE watchlist.source_type
                 END""",
            (ticker.upper(), company_name, themes_json, source_type, notes)
        )
        conn.commit()
        result = True
    except Exception:
        result = False
    conn.close()
    return result


def get_watchlist():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM watchlist ORDER BY added_at DESC"
    ).fetchall()
    conn.close()
    items = []
    for r in rows:
        item = dict(r)
        try:
            item["sub_themes"] = json.loads(item["sub_themes"]) if item["sub_themes"] else []
        except Exception:
            item["sub_themes"] = []
        items.append(item)
    return items


def update_watchlist_notes(ticker, notes):
    conn = get_connection()
    conn.execute("UPDATE watchlist SET notes = ? WHERE ticker = ?", (notes, ticker.upper()))
    conn.commit()
    conn.close()


def remove_from_watchlist(ticker):
    conn = get_connection()
    conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
    conn.commit()
    conn.close()


# ── WMA cache ─────────────────────────────────────────────────────────────────

def upsert_wma_cache(rows):
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    conn.executemany(
        """INSERT INTO wma_cache (ticker, company_name, index_name, current_price, wma_200, pct_from_wma, last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker) DO UPDATE SET
             company_name=excluded.company_name,
             index_name=excluded.index_name,
             current_price=excluded.current_price,
             wma_200=excluded.wma_200,
             pct_from_wma=excluded.pct_from_wma,
             last_updated=excluded.last_updated""",
        [(r["ticker"], r["company_name"], r["index_name"],
          r["current_price"], r["wma_200"], r["pct_from_wma"], now)
         for r in rows]
    )
    conn.commit()
    conn.close()


def get_wma_cache(below_only=True):
    conn = get_connection()
    query = "SELECT * FROM wma_cache"
    if below_only:
        query += " WHERE pct_from_wma < 0"
    query += " ORDER BY pct_from_wma ASC"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_wma_last_updated():
    conn = get_connection()
    row = conn.execute("SELECT MAX(last_updated) as ts FROM wma_cache").fetchone()
    conn.close()
    return row["ts"] if row else None
