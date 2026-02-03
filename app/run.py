from app.settings import ENROLLMENT_DIR, MODEL_PATH, DB_PATH, MIN_CONFIDENCE
from src.recognition.enrollment import enroll_all
from src.recognition.realtime import run_realtime
from src.storage.sqlite_db import connect


def is_enrollment_done(db_path):
    """Check if students are already enrolled"""
    conn = connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) FROM persons")
        count = cur.fetchone()[0]
    except Exception:
        count = 0

    conn.close()
    return count > 0


def main():
    """
    Entry point of AutoFace system
    - Enroll faces ONLY if not already enrolled
    - Run 30-second real-time attendance
    """

    # Step 1: Enrollment (runs only once automatically)
    if not is_enrollment_done(DB_PATH):
        print("🔹 Enrollment not found. Starting enrollment...")
        enroll_all(
            ENROLLMENT_DIR,
            MODEL_PATH,
            DB_PATH,
            min_confidence=MIN_CONFIDENCE
        )
        print("✅ Enrollment completed")
    else:
        print("✅ Enrollment already exists. Skipping enrollment")

    # Step 2: Real-time attendance (30 seconds, automatic)
    print("🎥 Starting real-time attendance (30 seconds)...")
    run_realtime()


if __name__ == "__main__":
    main()
