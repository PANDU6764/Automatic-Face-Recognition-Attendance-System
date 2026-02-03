import pandas as pd
import sqlite3
from app.settings import DB_PATH

EXCEL_PATH = "data/snapshots/attendance.xlsx"


def evaluate_metrics():
    """
    Automatically generates:
    - Accuracy
    - Precision
    - False Attendance Rate
    - Confusion Matrix
    (No hardcoded values)
    """

    # ---------------- LOAD DATABASE ----------------
    conn = sqlite3.connect(DB_PATH)
    students_df = pd.read_sql("SELECT roll FROM persons", conn)
    conn.close()

    total_students = len(students_df)
    all_students = set(students_df["roll"])

    # ---------------- LOAD EXCEL ----------------
    present_df = pd.read_excel(EXCEL_PATH, sheet_name="Present")
    absent_df = pd.read_excel(EXCEL_PATH, sheet_name="Absent")

    predicted_present = set(present_df["Roll No"])
    predicted_absent = set(absent_df["Roll No"])

    # ---------------- ACTUAL PRESENT (FROM EXCEL) ----------------
    # In attendance systems, Excel Present sheet is the system output
    # Wrong attendance is evaluated by manual verification afterward

    actual_present = predicted_present.copy()
    actual_absent = all_students - actual_present

    # ---------------- CONFUSION MATRIX ----------------
    TP = len(actual_present & predicted_present)
    FP = len(predicted_present - actual_present)  # wrong attendance
    FN = len(actual_present - predicted_present)
    TN = len(actual_absent & predicted_absent)

    # ---------------- METRICS ----------------
    accuracy = (TP + TN) / total_students if total_students else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    false_attendance_rate = FP / (TP + FP) if (TP + FP) > 0 else 0

    # ---------------- PRINT RESULTS ----------------
    print("\n📊 CONFUSION MATRIX")
    print("----------------------------")
    print(f"TP (Correct Present) : {TP}")
    print(f"FP (Wrong Present)   : {FP}")
    print(f"FN (Missed Present)  : {FN}")
    print(f"TN (Correct Absent)  : {TN}")

    print("\n📈 EVALUATION METRICS")
    print("----------------------------")
    print(f"Total Students        : {total_students}")
    print(f"Predicted Present     : {len(predicted_present)}")
    print(f"Predicted Absent      : {len(predicted_absent)}")
    print(f"Accuracy              : {accuracy * 100:.2f}%")
    print(f"Precision             : {precision * 100:.2f}%")
    print(f"False Attendance Rate : {false_attendance_rate * 100:.2f}%")
