import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = "pathoscope_data.db"

def init_db():
    """Creates the users and history tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Table for user accounts
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    # Table for scan history
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT, 
                  prediction TEXT, 
                  timestamp TEXT,
                  confidence REAL)''')
    conn.commit()
    conn.close()

def add_history(username, prediction, confidence):
    """Saves a scan result to the history."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO history (username, prediction, timestamp, confidence) VALUES (?, ?, ?, ?)", 
                  (username, prediction, now, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving history: {e}")

def get_user_history(username):
    """Retrieves all previous scans for a specific user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT prediction, timestamp, confidence FROM history WHERE username = ? ORDER BY timestamp DESC", (username,))
        data = c.fetchall()
        conn.close()
        return data
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []

# Keep your add_user and login_user functions from before...
