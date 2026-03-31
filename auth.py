"""
Authentication module for the AI Trading System.

Provides user registration, login, and session management using SQLite
and werkzeug password hashing.
"""

import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import config as cfg


def _get_db():
    """Get a SQLite connection with row_factory set."""
    conn = sqlite3.connect(cfg.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create the users table if it doesn't exist."""
    os.makedirs(os.path.dirname(cfg.DATABASE_PATH), exist_ok=True)
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def register_user(name, email, password):
    """
    Register a new user.

    Returns dict with 'success' and either 'user' or 'error'.
    """
    name = name.strip()
    email = email.strip().lower()

    if not name or len(name) < 2:
        return {'success': False, 'error': 'Name must be at least 2 characters'}
    if not email or '@' not in email or '.' not in email.split('@')[-1]:
        return {'success': False, 'error': 'Please enter a valid email address'}
    if not password or len(password) < 8:
        return {'success': False, 'error': 'Password must be at least 8 characters'}

    conn = _get_db()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            return {'success': False, 'error': 'An account with this email already exists'}

        password_hash = generate_password_hash(password)
        now = datetime.utcnow().isoformat()
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, now)
        )
        conn.commit()
        return {
            'success': True,
            'user': {
                'id': cursor.lastrowid,
                'name': name,
                'email': email,
                'created_at': now
            }
        }
    except sqlite3.IntegrityError:
        return {'success': False, 'error': 'An account with this email already exists'}
    finally:
        conn.close()


def authenticate_user(email, password):
    """
    Verify credentials.

    Returns user dict on success, None on failure.
    """
    email = email.strip().lower()
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        if row and check_password_hash(row['password_hash'], password):
            return {
                'id': row['id'],
                'name': row['name'],
                'email': row['email'],
                'created_at': row['created_at']
            }
        return None
    finally:
        conn.close()


def get_user_by_id(user_id):
    """Look up a user by their integer ID."""
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()
