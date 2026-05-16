"""
=============================================================================
  database_manager.py  
  AI-Driven Criminal Scam Analysis & Case Tracking System
=============================================================================
  PURPOSE  : Centralised file I/O layer.  Handles ALL reading from and
             writing to persistent storage:
               • crime_database.csv  — append-only case ledger
               • session_report.txt  — human-readable audit summary

  DESIGN   : Separation of concerns — app.py NEVER touches CSV/TXT files
             directly; it delegates every I/O operation to this module.
             This makes the code easier to test, audit, and extend.

  AUTHOR   : Ahmad Raza 
=============================================================================
"""

# ── Standard-library imports ──────────────────────────────────────────────
import os          # File existence checks
import csv         # Low-level CSV read / write
import uuid        # Generate globally unique Case IDs
import datetime    # ISO-8601 timestamps for every case record

# ── Third-party ───────────────────────────────────────────────────────────
import pandas as pd  # Higher-level CSV reading for search & analytics


# ─────────────────────────────────────────────────────────────────────────
#  CONFIGURATION — all paths in one place for maintainability
# ─────────────────────────────────────────────────────────────────────────
CRIME_DATABASE_PATH   = "crime_database.csv"  # Append-only case ledger
SESSION_REPORT_PATH   = "session_report.txt"  # Written at end of each audit session

# Ordered list of fields that EVERY case record must contain
DATABASE_FIELD_NAMES  = [
    "Case_ID",    # Unique identifier  (UUID4 hex)
    "Timestamp",  # ISO-8601 datetime  e.g. 2025-06-01 14:32:07
    "Message",    # Original evidence text submitted for analysis
    "Verdict",    # Human-readable result: SCAM / LEGITIMATE / SUSPICIOUS
    "Score",      # AI confidence score (probability of being scam, 0.00–1.00)
]

# Verdict thresholds — must stay in sync with app.py badge logic
SCAM_THRESHOLD        = 0.70   # P(scam) ≥ 0.70 → 🔴 Critical Scam
SUSPICIOUS_THRESHOLD  = 0.40   # 0.40 ≤ P(scam) < 0.70 → 🟡 Suspicious
                                # P(scam) < 0.40 → 🟢 Legitimate


# ─────────────────────────────────────────────────────────────────────────
#  HELPER — Generate a unique Case ID
#  WHY UUID4: Randomly generated, collision probability is astronomically
#             small.  Prefixing with 'AR-' makes it instantly identifiable
#             as originating from our forensics lab.
# ─────────────────────────────────────────────────────────────────────────
def generate_case_id() -> str:
    """Return a unique case identifier string, e.g. 'AR-A3F1B2C4'."""
    unique_hex_suffix = uuid.uuid4().hex[:8].upper()
    return f"AR-{unique_hex_suffix}"


# ─────────────────────────────────────────────────────────────────────────
#  HELPER — Determine verdict label from numeric score
# ─────────────────────────────────────────────────────────────────────────
def classify_scam_verdict(scam_probability: float) -> str:
    """
    Map a raw AI probability score to a human-readable forensic verdict.

    Args:
        scam_probability (float): P(scam) output from model.predict_proba()

    Returns:
        str: One of 'CRITICAL SCAM', 'SUSPICIOUS', 'LEGITIMATE'
    """
    if scam_probability >= SCAM_THRESHOLD:
        return "CRITICAL SCAM"
    elif scam_probability >= SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS"
    else:
        return "LEGITIMATE"


# ─────────────────────────────────────────────────────────────────────────
#  CORE WRITE OPERATION — Append a single case to the CSV ledger
#  WHY append mode: We never overwrite previous evidence.  In forensics,
#  data integrity means the ledger only grows, never shrinks.
# ─────────────────────────────────────────────────────────────────────────
def save_case_to_database(
    original_message : str,
    ai_verdict       : str,
    confidence_score : float,
) -> str:
    """
    Append one case record to 'crime_database.csv' and return its Case ID.

    If the CSV does not yet exist, this function creates it with a header row.

    Args:
        original_message (str) : The raw evidence text that was analysed.
        ai_verdict       (str) : Human-readable verdict string.
        confidence_score (float): P(scam) probability from the AI model.

    Returns:
        str: The Case ID assigned to this record (e.g. 'AR-A3F1B2C4').
    """
    # Generate unique identifiers for this record
    assigned_case_id   = generate_case_id()
    current_timestamp  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build the full record as a Python dict for clarity
    case_record = {
        "Case_ID"   : assigned_case_id,
        "Timestamp" : current_timestamp,
        "Message"   : original_message.replace("\n", " "),  # Flatten multi-line input
        "Verdict"   : ai_verdict,
        "Score"     : f"{confidence_score:.4f}",            # 4 decimal places
    }

    try:
        # Determine whether we are creating the file (needs header) or appending
        file_already_exists = os.path.exists(CRIME_DATABASE_PATH)

        # Open in append mode — 'a' never truncates existing data
        with open(CRIME_DATABASE_PATH, "a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=DATABASE_FIELD_NAMES)

            # Write the header only when creating the file for the first time
            if not file_already_exists:
                csv_writer.writeheader()

            # Write the actual case record
            csv_writer.writerow(case_record)

    except IOError as io_error:
        # Surface the error to the caller so the UI can display a warning
        raise IOError(
            f"[DATABASE ERROR] Could not write to '{CRIME_DATABASE_PATH}': {io_error}"
        )

    return assigned_case_id


# ─────────────────────────────────────────────────────────────────────────
#  CORE READ OPERATION — Load all cases from the CSV ledger
# ─────────────────────────────────────────────────────────────────────────
def load_all_case_records() -> pd.DataFrame:
    """
    Read and return the entire crime_database.csv as a pandas DataFrame.

    Returns an empty DataFrame (with correct columns) if the file does not
    exist yet, so callers never have to guard against None.
    """
    if not os.path.exists(CRIME_DATABASE_PATH):
        # Return an empty but correctly structured DataFrame
        return pd.DataFrame(columns=DATABASE_FIELD_NAMES)

    try:
        all_cases_dataframe = pd.read_csv(CRIME_DATABASE_PATH, encoding="utf-8")
        return all_cases_dataframe

    except pd.errors.EmptyDataError:
        # File exists but has no rows (edge case: only header written)
        return pd.DataFrame(columns=DATABASE_FIELD_NAMES)

    except Exception as read_error:
        raise IOError(
            f"[DATABASE ERROR] Failed to read '{CRIME_DATABASE_PATH}': {read_error}"
        )


# ─────────────────────────────────────────────────────────────────────────
#  SEARCH OPERATION — Filter cases by keyword in the Message field
#  WHY: Investigators often need to cross-reference a suspect phrase
#       (e.g., "click here", "bank account") across historical cases.
# ─────────────────────────────────────────────────────────────────────────
def search_case_records_by_keyword(search_keyword: str) -> pd.DataFrame:
    """
    Search the crime database for rows whose Message contains the keyword.

    The search is case-insensitive to avoid missing records due to
    capitalisation differences.

    Args:
        search_keyword (str): The string to search for within the Message field.

    Returns:
        pd.DataFrame: Filtered rows, or an empty DataFrame if nothing found.
    """
    all_cases_dataframe = load_all_case_records()

    # Empty database — nothing to search
    if all_cases_dataframe.empty:
        return all_cases_dataframe

    # Ensure Message column is treated as string (guards against NaN values)
    message_column = all_cases_dataframe["Message"].fillna("").astype(str)

    # Perform case-insensitive substring match
    keyword_lower   = search_keyword.strip().lower()
    match_mask      = message_column.str.lower().str.contains(
        keyword_lower, regex=False   # regex=False for literal string matching
    )

    matching_records = all_cases_dataframe[match_mask].copy()
    return matching_records


# ─────────────────────────────────────────────────────────────────────────
#  SEARCH BY CASE ID — Retrieve a single record by its unique ID
# ─────────────────────────────────────────────────────────────────────────
def fetch_case_by_id(target_case_id: str) -> pd.DataFrame:
    """
    Retrieve the case record matching the given Case_ID.

    Args:
        target_case_id (str): Case ID string, e.g. 'AR-A3F1B2C4'.

    Returns:
        pd.DataFrame: Single-row DataFrame, or empty if not found.
    """
    all_cases_dataframe = load_all_case_records()

    if all_cases_dataframe.empty:
        return all_cases_dataframe

    matching_records = all_cases_dataframe[
        all_cases_dataframe["Case_ID"] == target_case_id.strip().upper()
    ]
    return matching_records


# ─────────────────────────────────────────────────────────────────────────
#  SESSION REPORT WRITER
#  WHY: After each investigative session, a plain-text audit trail must be
#       written.  Unlike the CSV (which accumulates forever), the session
#       report is overwritten each run — it summarises ONLY that session.
# ─────────────────────────────────────────────────────────────────────────
def write_session_report(session_statistics: dict) -> None:
    """
    Write a formatted forensic session report to 'session_report.txt'.

    Args:
        session_statistics (dict): Keys expected:
            - session_start    (str) : ISO datetime when session began
            - session_end      (str) : ISO datetime when report was generated
            - total_analysed   (int) : Number of messages scanned
            - scam_count       (int) : Count of CRITICAL SCAM verdicts
            - suspicious_count (int) : Count of SUSPICIOUS verdicts
            - legitimate_count (int) : Count of LEGITIMATE verdicts
            - case_ids         (list): All Case IDs generated this session
    """
    report_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build the report as a multi-line string using an f-string block
    report_content = f"""
================================================================================
        AR Forensics & CyberSecurity Labs — OFFICIAL SESSION AUDIT REPORT
                      Criminal Scam Analysis System v1.0
================================================================================

  REPORT GENERATED  : {report_timestamp}
  SESSION START     : {session_statistics.get('session_start', 'N/A')}
  SESSION END       : {session_statistics.get('session_end', 'N/A')}

--------------------------------------------------------------------------------
                            ANALYSIS SUMMARY
--------------------------------------------------------------------------------

  Total Messages Analysed  : {session_statistics.get('total_analysed', 0)}
  ┌──────────────────────────────────────────┐
  │  🔴  Critical Scam   : {session_statistics.get('scam_count', 0):<5}              │
  │  🟡  Suspicious      : {session_statistics.get('suspicious_count', 0):<5}              │
  │  🟢  Legitimate      : {session_statistics.get('legitimate_count', 0):<5}              │
  └──────────────────────────────────────────┘

--------------------------------------------------------------------------------
                             CASE REGISTER
--------------------------------------------------------------------------------

{chr(10).join(f'  [{i+1:02d}]  {cid}' for i, cid in enumerate(session_statistics.get('case_ids', [])))}

--------------------------------------------------------------------------------
                            INTEGRITY STATEMENT
--------------------------------------------------------------------------------

  This report was automatically generated by the AR Forensics & CyberSecurity Labs
  — Criminal Scam Analysis System.All case records are stored in 'crime_database.csv'\n
  and may be retrieved at any time for further forensic examination.

  Analyst Signature : ______________________
  Badge / ID        : ______________________
  Date              : ______________________

================================================================================
             AR Forensics & CyberSecurity Labs — CONFIDENTIAL
================================================================================
"""

    try:
        # 'w' mode intentionally overwrites — session report is per-session
        with open(SESSION_REPORT_PATH, "w", encoding="utf-8") as report_file:
            report_file.write(report_content.strip())
    except IOError as io_error:
        raise IOError(
            f"[REPORT ERROR] Could not write to '{SESSION_REPORT_PATH}': {io_error}"
        )


# ─────────────────────────────────────────────────────────────────────────
#  SUMMARY STATISTICS — Compute verdicts distribution from the live database
# ─────────────────────────────────────────────────────────────────────────
def compute_database_statistics() -> dict:
    """
    Return a summary dict for dashboard metrics and charts.

    Returns:
        dict: {
            'total'       : int,
            'scam'        : int,
            'suspicious'  : int,
            'legitimate'  : int,
        }
    """
    all_cases = load_all_case_records()

    if all_cases.empty:
        return {"total": 0, "scam": 0, "suspicious": 0, "legitimate": 0}

    verdict_counts = all_cases["Verdict"].value_counts()

    return {
        "total"      : len(all_cases),
        "scam"       : int(verdict_counts.get("CRITICAL SCAM", 0)),
        "suspicious" : int(verdict_counts.get("SUSPICIOUS",    0)),
        "legitimate" : int(verdict_counts.get("LEGITIMATE",    0)),
    }


# ─────────────────────────────────────────────────────────────────────────
#  QUICK SELF-TEST — run this file directly to verify the I/O layer works
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running database_manager self-test …\n")

    # Test 1: Save a dummy case
    test_case_id = save_case_to_database(
        original_message = "Congratulations! You won a free prize. Call now.",
        ai_verdict       = "CRITICAL SCAM",
        confidence_score = 0.9231,
    )
    print(f"  ✓ Saved test case  →  {test_case_id}")

    # Test 2: Save a legitimate case
    legit_case_id = save_case_to_database(
        original_message = "Hey, are you coming to the lecture tomorrow?",
        ai_verdict       = "LEGITIMATE",
        confidence_score = 0.0412,
    )
    print(f"  ✓ Saved legit case →  {legit_case_id}")

    # Test 3: Load all records
    all_records = load_all_case_records()
    print(f"\n  ✓ Total records in database: {len(all_records)}")

    # Test 4: Search by keyword
    search_results = search_case_records_by_keyword("congratulations")
    print(f"  ✓ Search 'congratulations': {len(search_results)} result(s)")

    # Test 5: Write a session report
    write_session_report({
        "session_start"    : "2025-06-01 09:00:00",
        "session_end"      : "2025-06-01 09:05:00",
        "total_analysed"   : 2,
        "scam_count"       : 1,
        "suspicious_count" : 0,
        "legitimate_count" : 1,
        "case_ids"         : [test_case_id, legit_case_id],
    })
    print(f"\n  ✓ Session report written  →  '{SESSION_REPORT_PATH}'")
    print("\n  All self-tests passed ✓")
