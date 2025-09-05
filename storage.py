import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'data.db')

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            created_at TEXT,
            status TEXT,
            total_amount REAL,
            items_json TEXT,
            raw_request TEXT,
            response_text TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            role TEXT,
            text TEXT,
            order_id INTEGER NULL,
            FOREIGN KEY(order_id) REFERENCES orders(id)
        )""")
        conn.commit()

def save_order(order_id:int, status:str, items:list, raw_request:str, response_text:str, total:float):
    with get_conn() as conn:
        conn.execute("REPLACE INTO orders (id, created_at, status, total_amount, items_json, raw_request, response_text) VALUES (?,?,?,?,?,?,?)",
                     (order_id, datetime.utcnow().isoformat(), status, total, json.dumps(items), raw_request, response_text))
        conn.commit()

def update_order_status(order_id:int, status:str):
    with get_conn() as conn:
        conn.execute("UPDATE orders SET status=? WHERE id=?", (status, order_id))
        conn.commit()

def load_orders():
    with get_conn() as conn:
        rows = conn.execute("SELECT id, status, total_amount, items_json FROM orders ORDER BY id ASC").fetchall()
    orders = []
    for rid, status, total, items_json in rows:
        try:
            items = json.loads(items_json)
        except Exception:
            items = []
        tuple_items = [(i['name'], i['qty']) for i in items]
        orders.append({"id": rid, "status": status, "total_amount": total, "items": tuple_items})
    return orders

def save_chat(role:str, text:str, order_id=None):
    with get_conn() as conn:
        conn.execute("INSERT INTO chat_messages (ts, role, text, order_id) VALUES (?,?,?,?)",
                     (datetime.utcnow().isoformat(), role, text, order_id))
        conn.commit()

def load_chat(limit:int=200):
    with get_conn() as conn:
        rows = conn.execute("SELECT ts, role, text, IFNULL(order_id,'') FROM chat_messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    messages = []
    for ts, role, text, oid in rows:
        messages.append({"ts": ts, "role": role, "text": text, "order_id": oid if oid!='' else None})
    return list(reversed(messages))

def price_for_item(name:str) -> float:
    # Updated prices to match INVENTORY
    base = {
        'milk': 25.0,
        'bread': 35.0,
        'rice': 80.0,
        'maggi': 15.0
    }
    return base.get(name, 10.0)
import os
import sqlite3
import json
import threading
import queue
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'data.db')

_order_queue = queue.Queue()
_order_thread_started = False
_lock = threading.Lock()

PRICES = {
    'milk': 25.0,
    'bread': 35.0,
    'rice': 80.0,
    'maggi': 15.0,
}

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            created_at TEXT,
            status TEXT,
            total_amount REAL,
            raw_request TEXT,
            response_text TEXT,
            items_json TEXT
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER,
            item_name TEXT,
            qty INTEGER,
            unit_price REAL,
            line_total REAL,
            FOREIGN KEY(order_id) REFERENCES orders(id)
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            role TEXT,
            text TEXT,
            order_id INTEGER NULL,
            FOREIGN KEY(order_id) REFERENCES orders(id)
        )
        """)
        conn.commit()

def start_order_worker():
    global _order_thread_started
    with _lock:
        if _order_thread_started:
            return
        t = threading.Thread(target=_order_worker, daemon=True, name='OrderSaverThread')
        t.start()
        _order_thread_started = True

def _order_worker():
    while True:
        order_data = _order_queue.get()
        try:
            _persist_order(order_data)
        except Exception:
            # In production, log properly
            pass
        finally:
            _order_queue.task_done()

def enqueue_order(order_data: dict):
    _order_queue.put(order_data)

def _persist_order(order_data: dict):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO orders (id, created_at, status, total_amount, raw_request, response_text, items_json) VALUES (?,?,?,?,?,?,?)",
            (
                order_data['id'],
                order_data['created_at'],
                order_data['status'],
                order_data['total_amount'],
                order_data.get('raw_request',''),
                order_data.get('response_text',''),
                json.dumps(order_data['items'])
            )
        )
        for item in order_data['items']:
            name = item['name']
            qty = item['qty']
            up = item['unit_price']
            lt = item['line_total']
            c.execute(
                "INSERT INTO order_items (order_id, item_name, qty, unit_price, line_total) VALUES (?,?,?,?,?)",
                (order_data['id'], name, qty, up, lt)
            )
        conn.commit()

def update_order_status(order_id: int, new_status: str):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("UPDATE orders SET status=? WHERE id=?", (new_status, order_id))
        conn.commit()

def save_chat(role: str, text: str, order_id=None):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_messages (ts, role, text, order_id) VALUES (?,?,?,?)",
            (datetime.utcnow().isoformat(), role, text, order_id)
        )
        conn.commit()

def load_orders():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id, status, items_json, total_amount FROM orders ORDER BY id ASC")
        rows = c.fetchall()
    orders = []
    for rid, status, items_json, total in rows:
        try:
            items = json.loads(items_json)
        except Exception:
            items = []
        # Convert items into list of tuples for compatibility
        tuple_items = [(i['name'], i['qty']) for i in items]
        orders.append({"id": rid, "items": tuple_items, "status": status, "total_amount": total})
    return orders

def price_for_item(name: str) -> float:
    return PRICES.get(name, 10.0)
