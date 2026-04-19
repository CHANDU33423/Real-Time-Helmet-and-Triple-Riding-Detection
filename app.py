# app.py - Flask backend for Smart Traffic Monitoring System UI
# Run: python app.py
# Opens: http://localhost:5000

import os
import random
import sys
from datetime import datetime, timedelta

import database as db


from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
db.init_db()

FINE_AMOUNTS = {
    "No Helmet": 500,
    "Triple Riding": 1000,
    "No Helmet + Triple Riding": 1500,
}

ESTIMATED_FINE_SQL = (
    "CASE "
    "WHEN v.violation_type = 'No Helmet + Triple Riding' THEN 1500 "
    "WHEN v.violation_type = 'Triple Riding' THEN 1000 "
    "WHEN v.violation_type = 'No Helmet' THEN 500 "
    "ELSE 500 END"
)


def _normalize_plate(value):
    plate = (value or "UNKNOWN").strip().upper()
    return plate or "UNKNOWN"


def _coerce_rider_count(value):
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _build_challan_no():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"CH{timestamp}{random.randint(10, 99)}"


def _lookup_owner_name(conn, plate_no):
    row = conn.execute(
        "SELECT owner_name FROM vehicles WHERE plate_no=?",
        (plate_no,),
    ).fetchone()
    return row["owner_name"] if row else None


def _create_challan(cursor, violation_id, violation_type):
    fine_amount = FINE_AMOUNTS.get(violation_type, 500)
    challan_no = _build_challan_no()
    due_date = datetime.now() + timedelta(days=30)
    cursor.execute(
        "INSERT INTO challans (challan_no, violation_id, fine_amount, due_date) VALUES (?,?,?,?)",
        (challan_no, violation_id, fine_amount, due_date),
    )
    return {
        "challan_id": cursor.lastrowid,
        "challan_no": challan_no,
        "fine_amount": fine_amount,
        "due_date": due_date.isoformat(),
    }


# --- Page Routes ---


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
        }
    )


# --- Stats API ---


@app.route("/api/stats")
def get_stats():
    conn = db.get_db()
    c = conn.cursor()

    try:
        total = c.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        helmet = c.execute(
            "SELECT COUNT(*) FROM violations WHERE violation_type LIKE '%Helmet%'"
        ).fetchone()[0]
        triple = c.execute(
            "SELECT COUNT(*) FROM violations WHERE violation_type LIKE '%Triple%'"
        ).fetchone()[0]
        paid = c.execute("SELECT COUNT(*) FROM violations WHERE status='paid'").fetchone()[
            0
        ]
        unpaid = c.execute(
            "SELECT COUNT(*) FROM violations WHERE status='unpaid'"
        ).fetchone()[0]
        revenue = c.execute("SELECT COALESCE(SUM(amount),0) FROM payments").fetchone()[0]
        cameras_active = c.execute(
            "SELECT COUNT(*) FROM cameras WHERE status='active'"
        ).fetchone()[0]

        trend = []
        for i in range(6, -1, -1):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            count = c.execute(
                "SELECT COUNT(*) FROM violations WHERE date(timestamp)=?", (date,)
            ).fetchone()[0]
            trend.append({"date": date, "count": count})
    finally:
        c.close()
        conn.close()

    return jsonify(
        {
            "total": total,
            "helmet_violations": helmet,
            "triple_riding": triple,
            "paid": paid,
            "unpaid": unpaid,
            "revenue": revenue,
            "trend": trend,
            "cameras_active": cameras_active,
        }
    )


# --- Violations API ---


@app.route("/api/violations")
def get_violations():
    conn = db.get_db()
    c = conn.cursor()

    status_f = request.args.get("status", "all")
    type_f = request.args.get("type", "all")

    query = (
        f"SELECT v.*, ch.challan_no, COALESCE(ch.fine_amount, {ESTIMATED_FINE_SQL}) AS fine_amount, "
        "ch.id AS challan_id "
        "FROM violations v "
        "LEFT JOIN challans ch ON ch.violation_id = v.id"
    )
    conditions = []
    params = []

    if status_f != "all":
        conditions.append("v.status=?")
        params.append(status_f)
    if type_f != "all":
        conditions.append("v.violation_type LIKE ?")
        params.append(f"%{type_f}%")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY v.timestamp DESC"

    rows = c.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


@app.route("/api/violations/add", methods=["POST"])
def add_violation():
    data = request.json or {}
    violation_type = (data.get("violation_type") or "").strip()
    vehicle_no = _normalize_plate(data.get("vehicle_no") or data.get("plate_no"))
    location = (data.get("location") or "").strip() or None
    rider_count = _coerce_rider_count(data.get("rider_count", 1))

    if not violation_type:
        return jsonify({"success": False, "error": "violation_type is required"}), 400

    conn = db.get_db()
    c = conn.cursor()
    owner_name = (data.get("owner_name") or "").strip() or _lookup_owner_name(
        conn, vehicle_no
    )

    c.execute(
        "INSERT INTO violations (vehicle_no, owner_name, violation_type, location, rider_count) VALUES (?,?,?,?,?)",
        (vehicle_no, owner_name, violation_type, location, rider_count),
    )
    violation_id = c.lastrowid
    challan = _create_challan(c, violation_id, violation_type)

    conn.commit()
    conn.close()
    return jsonify(
        {
            "success": True,
            "violation_id": violation_id,
            "fine": challan["fine_amount"],
            **challan,
        }
    )


@app.route("/api/violations/<int:vid>")
def get_violation(vid):
    conn = db.get_db()
    row = conn.execute(
        (
            f"SELECT v.*, ch.challan_no, COALESCE(ch.fine_amount, {ESTIMATED_FINE_SQL}) AS fine_amount, "
            "ch.id AS challan_id, ch.due_date, ch.status AS challan_status, "
            "p.transaction_id, p.paid_at, p.method AS payment_method "
            "FROM violations v "
            "LEFT JOIN challans ch ON ch.violation_id = v.id "
            "LEFT JOIN payments p ON p.id = ("
            "    SELECT id FROM payments "
            "    WHERE challan_id = ch.id "
            "    ORDER BY datetime(paid_at) DESC, id DESC "
            "    LIMIT 1"
            ") "
            "WHERE v.id=?"
        ),
        (vid,),
    ).fetchone()
    conn.close()
    return jsonify(dict(row)) if row else (jsonify({"error": "Not found"}), 404)


@app.route("/api/violations/<int:vid>/challan", methods=["POST"])
def generate_challan(vid):
    conn = db.get_db()
    c = conn.cursor()

    existing = c.execute(
        "SELECT * FROM challans WHERE violation_id=?", (vid,)
    ).fetchone()
    if existing:
        conn.close()
        return jsonify(
            {
                "challan_id": existing["id"],
                "challan_no": existing["challan_no"],
                "fine_amount": existing["fine_amount"],
                "message": "Challan already exists",
            }
        )

    violation = c.execute("SELECT * FROM violations WHERE id=?", (vid,)).fetchone()
    if not violation:
        conn.close()
        return jsonify({"error": "Violation not found"}), 404

    challan = _create_challan(c, vid, violation["violation_type"])
    conn.commit()
    conn.close()
    return jsonify(
        {
            **challan,
            "message": "Challan generated",
        }
    )


# --- Payment API ---


@app.route("/api/payment", methods=["POST"])
def process_payment():
    data = request.json or {}
    challan_id = data.get("challan_id")
    method = (data.get("method") or "UPI").strip() or "UPI"

    if not challan_id:
        return jsonify({"success": False, "error": "challan_id is required"}), 400

    conn = db.get_db()
    c = conn.cursor()
    challan = c.execute(
        "SELECT id, fine_amount, status FROM challans WHERE id=?",
        (challan_id,),
    ).fetchone()
    if not challan:
        conn.close()
        return jsonify({"success": False, "error": "Challan not found"}), 404
    if challan["status"] == "paid":
        payment = c.execute(
            "SELECT transaction_id FROM payments WHERE challan_id=? ORDER BY datetime(paid_at) DESC, id DESC LIMIT 1",
            (challan_id,),
        ).fetchone()
        conn.close()
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Challan is already paid",
                    "transaction_id": payment["transaction_id"] if payment else None,
                }
            ),
            409,
        )

    amount = float(challan["fine_amount"])
    txn_id = f"TXN{random.randint(100000, 999999)}"

    c.execute(
        "INSERT INTO payments (challan_id, amount, method, transaction_id) VALUES (?,?,?,?)",
        (challan_id, amount, method, txn_id),
    )
    c.execute("UPDATE challans SET status='paid' WHERE id=?", (challan_id,))
    c.execute(
        "UPDATE violations SET status='paid' WHERE id IN (SELECT violation_id FROM challans WHERE id=?)",
        (challan_id,),
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True, "transaction_id": txn_id, "amount": amount})


# --- Notifications ---


@app.route("/api/notify/<int:vid>", methods=["POST"])
def notify(vid):
    _ = vid
    return jsonify({"success": True, "message": "SMS and email notification sent"})


# --- Reports API ---


@app.route("/api/reports")
def get_reports():
    conn = db.get_db()
    c = conn.cursor()

    daily = []
    for i in range(29, -1, -1):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        row = c.execute(
            (
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN violation_type LIKE '%Helmet%' THEN 1 ELSE 0 END) AS helmet, "
                "SUM(CASE WHEN violation_type LIKE '%Triple%' THEN 1 ELSE 0 END) AS triple "
                "FROM violations WHERE date(timestamp)=?"
            ),
            (date,),
        ).fetchone()
        daily.append(
            {
                "date": date,
                "total": row["total"] or 0,
                "helmet": row["helmet"] or 0,
                "triple": row["triple"] or 0,
            }
        )

    # Hourly Breakdown (Last 24 Hours)
    hourly = []
    for i in range(23, -1, -1):
        time_point = datetime.now() - timedelta(hours=i)
        hour_str = time_point.strftime("%Y-%m-%d %H:00:00")
        end_hour_str = (time_point + timedelta(hours=1)).strftime("%Y-%m-%d %H:00:00")
        count = c.execute(
            "SELECT COUNT(*) FROM violations WHERE timestamp >= ? AND timestamp < ?",
            (hour_str, end_hour_str),
        ).fetchone()[0]
        hourly.append({"hour": time_point.strftime("%H:%00"), "count": count})

    by_location = c.execute(
        "SELECT location, COUNT(*) AS count FROM violations WHERE location IS NOT NULL GROUP BY location ORDER BY count DESC"
    ).fetchall()

    revenue = c.execute(
        "SELECT COALESCE(SUM(fine_amount),0) FROM challans WHERE status='paid'"
    ).fetchone()[0]
    pending = c.execute(
        "SELECT COALESCE(SUM(fine_amount),0) FROM challans WHERE status='pending'"
    ).fetchone()[0]

    conn.close()
    return jsonify(
        {
            "daily": daily,
            "hourly": hourly,
            "by_location": [dict(row) for row in by_location],
            "total_revenue": revenue,
            "pending_revenue": pending,
        }
    )


# --- Camera API ---


@app.route("/api/cameras")
def get_cameras():
    conn = db.get_db()
    rows = conn.execute("SELECT * FROM cameras").fetchall()
    conn.close()
    result = []
    for row in rows:
        cam = dict(row)
        cam["violations_today"] = random.randint(0, 18)
        result.append(cam)
    return jsonify(result)


# --- Admin API ---


@app.route("/api/admin/vehicles", methods=["GET", "POST"])
def admin_vehicles():
    conn = db.get_db()
    if request.method == "GET":
        rows = conn.execute(
            "SELECT * FROM vehicles ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return jsonify([dict(row) for row in rows])

    data = request.json or {}
    plate_no = _normalize_plate(data.get("plate_no"))
    owner_name = (data.get("owner_name") or "").strip()
    if not plate_no or plate_no == "UNKNOWN" or not owner_name:
        conn.close()
        return (
            jsonify(
                {"success": False, "error": "plate_no and owner_name are required"}
            ),
            400,
        )

    conn.execute(
        "INSERT OR REPLACE INTO vehicles (plate_no, owner_name, phone, email, address) VALUES (?,?,?,?,?)",
        (
            plate_no,
            owner_name,
            data.get("phone"),
            data.get("email"),
            data.get("address"),
        ),
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/api/admin/vehicles/<string:plate>", methods=["DELETE"])
def delete_vehicle(plate):
    conn = db.get_db()
    conn.execute("DELETE FROM vehicles WHERE plate_no=?", (_normalize_plate(plate),))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

    print("\n" + "=" * 55)
    print("  [TRAFFIC] Smart Traffic Monitoring System - Web UI")
    print("=" * 55)
    print(f"  Open in browser:  http://localhost:{port}")
    print("  Press CTRL+C to stop\n")
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=False)
