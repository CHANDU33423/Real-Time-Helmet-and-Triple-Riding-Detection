"""
notification_service.py — Dispatcher for violation alerts (Email/SMS)
"""

import os
import threading
from datetime import datetime

# --- Configuration ---
# In a real app, use environment variables or a secure vault
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("TRAFFIC_SENDER_EMAIL", "alerts@traffic-system.gov")
SENDER_PWD = os.getenv("TRAFFIC_SENDER_PWD", "")

# --- Service Class ---


class NotificationService:
    @staticmethod
    def send_violation_alert(v_type: str, plate: str, location: str):
        """Dispatches alerts via background threads to prevent blocking detection."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg_body = (
            f"🚨 TRAFFIC VIOLATION DETECTED 🚨\n\n"
            f"Type:     {v_type}\n"
            f"Plate:    {plate}\n"
            f"Location: {location}\n"
            f"Time:     {ts}\n\n"
            f"Digital e-challan has been generated and sent to the owner."
        )

        # 1. Log to console & file (Simulation Mode)
        print(f"\n[NOTIF] Dispatching alert: {v_type} | {plate} | {location}")
        _log_to_file(msg_body)

        # 2. Attempt SMTP if configured
        if SENDER_EMAIL and SENDER_PWD:
            threading.Thread(target=_send_email, args=(msg_body,)).start()

        # 3. Simulate SMS
        print(f"[NOTIF] SMS sent to owner of {plate} ✅")


def _log_to_file(content: str):
    log_path = os.path.join("output", "logs", "notification_history.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write("-" * 40 + "\n")
        f.write(content + "\n")


def _send_email(body: str):
    try:
        # Placeholder for real SMTP logic
        # server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        # server.starttls()
        # server.login(SENDER_EMAIL, SENDER_PWD)
        # ... logic to find owner email from DB ...
        pass
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
