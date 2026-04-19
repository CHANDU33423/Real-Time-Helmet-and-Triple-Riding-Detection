"""
database.py — SQLite database setup and seeding for Traffic Monitoring System
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic_system.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            ip_address TEXT,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_no TEXT UNIQUE NOT NULL,
            owner_name TEXT,
            phone TEXT,
            email TEXT,
            address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_no TEXT NOT NULL,
            owner_name TEXT,
            violation_type TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            location TEXT,
            evidence_image TEXT,
            camera_id INTEGER,
            rider_count INTEGER DEFAULT 1,
            status TEXT DEFAULT 'unpaid',
            FOREIGN KEY (camera_id) REFERENCES cameras(id)
        );

        CREATE TABLE IF NOT EXISTS challans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            challan_no TEXT UNIQUE,
            violation_id INTEGER NOT NULL,
            fine_amount REAL NOT NULL,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            due_date TIMESTAMP,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (violation_id) REFERENCES violations(id)
        );

        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            challan_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            method TEXT,
            transaction_id TEXT,
            paid_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'success',
            FOREIGN KEY (challan_id) REFERENCES challans(id)
        );
    """)

    if c.execute("SELECT COUNT(*) FROM cameras").fetchone()[0] == 0:
        _seed(c)

    conn.commit()
    conn.close()


def _seed(c):
    # Cameras
    cameras = [
        ("Camera 01", "NH-44 Main Junction", "active", "192.168.1.101"),
        ("Camera 02", "Rajiv Gandhi Salai", "active", "192.168.1.102"),
        ("Camera 03", "Anna Salai Flyover", "inactive", "192.168.1.103"),
        ("Camera 04", "OMR Toll Plaza", "active", "192.168.1.104"),
    ]
    c.executemany(
        "INSERT INTO cameras (name, location, status, ip_address) VALUES (?,?,?,?)",
        cameras,
    )

    # Vehicles
    vehicles = [
        (
            "TN01AB1234",
            "Ravi Kumar",
            "9876543210",
            "ravi@email.com",
            "12, Anna Nagar, Chennai",
        ),
        (
            "TN02XY5678",
            "Priya Sharma",
            "9123456789",
            "priya@email.com",
            "45, T Nagar, Chennai",
        ),
        (
            "TN05MN9012",
            "Arjun Singh",
            "9988776655",
            "arjun@email.com",
            "78, Adyar, Chennai",
        ),
        (
            "TN07PQ3456",
            "Deepa Rajan",
            "9871234567",
            "deepa@email.com",
            "23, Velachery, Chennai",
        ),
        (
            "TN09RS7890",
            "Karthik Raj",
            "9765432109",
            "karthik@email.com",
            "56, Ambattur, Chennai",
        ),
        (
            "TN11UV2345",
            "Meena Devi",
            "9654321098",
            "meena@email.com",
            "89, Porur, Chennai",
        ),
        (
            "TN13WX6789",
            "Suresh Kumar",
            "9543210987",
            "suresh@email.com",
            "34, Tambaram, Chennai",
        ),
        (
            "TN15YZ0123",
            "Lakshmi Iyer",
            "9432109876",
            "lakshmi@email.com",
            "67, Guindy, Chennai",
        ),
    ]
    c.executemany(
        "INSERT INTO vehicles (plate_no, owner_name, phone, email, address) VALUES (?,?,?,?,?)",
        vehicles,
    )

    plates = [v[0] for v in vehicles]
    owners = [v[1] for v in vehicles]
    vtypes = ["No Helmet", "Triple Riding", "No Helmet + Triple Riding"]
    locs = [
        "NH-44 Main Junction",
        "Rajiv Gandhi Salai",
        "Anna Salai Flyover",
        "OMR Toll Plaza",
    ]
    fines = {"No Helmet": 500, "Triple Riding": 1000, "No Helmet + Triple Riding": 1500}
    now = datetime.now()

    vid_list = []
    for i in range(20):
        idx = i % len(plates)
        vt = vtypes[i % 3]
        ts = now - timedelta(
            days=random.randint(0, 6),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        st = "paid" if i < 12 else "unpaid"
        riders = 3 if "Triple" in vt else 1

        c.execute(
            """INSERT INTO violations (vehicle_no, owner_name, violation_type, timestamp, location, rider_count, status, camera_id)
               VALUES (?,?,?,?,?,?,?,?)""",
            (plates[idx], owners[idx], vt, ts, locs[i % 4], riders, st, (i % 4) + 1),
        )
        vid_list.append(c.lastrowid)

    for i, vid in enumerate(vid_list):
        vt = vtypes[i % 3]
        fine = fines[vt]
        challan_no = f"CH{2024001 + i:07d}"
        due = now + timedelta(days=30)
        cs = "paid" if i < 12 else "pending"

        c.execute(
            "INSERT INTO challans (challan_no, violation_id, fine_amount, due_date, status) VALUES (?,?,?,?,?)",
            (challan_no, vid, fine, due, cs),
        )

        if i < 12:
            cid = c.lastrowid
            c.execute(
                "INSERT INTO payments (challan_id, amount, method, transaction_id) VALUES (?,?,?,?)",
                (
                    cid,
                    fine,
                    ["UPI", "Card", "Net Banking"][i % 3],
                    f"TXN{random.randint(100000, 999999)}",
                ),
            )
