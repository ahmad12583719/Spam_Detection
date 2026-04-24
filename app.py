"""
=============================================================================
  app.py  
  AI-Driven Criminal Scam Analysis & Case Tracking System
=============================================================================
  PURPOSE  : Main Streamlit application — the investigator's interface.

  PAGES    : • Dashboard    – live KPI cards + visual analytics
             • New Analysis – submit evidence text for AI classification
             • Crime Records – search, browse, and export case history

  RUN WITH : streamlit run app.py

  AUTHOR   : Ahmad Raza 
=============================================================================
"""

# ── Standard-library ──────────────────────────────────────────────────────
import os
import datetime

# ── Third-party: UI & visualisation ──────────────────────────────────────
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

# ── Project modules ───────────────────────────────────────────────────────
import database_manager as db
from pdf_report_generator import generate_forensic_pdf_report
from model_trainer import clean_crime_evidence   # Reuse the same cleaning fn

# ─────────────────────────────────────────────────────────────────────────
#  PAGE-LEVEL STREAMLIT CONFIG — must be the first st.* call
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AR Forensics & CyberSecurity Labs",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS INJECTION
#  WHY: Streamlit's default theme is generic.  We inject CSS to create a
#       dark, professional forensics-lab aesthetic matching the PDF reports.
# ─────────────────────────────────────────────────────────────────────────
CUSTOM_STYLES = """
<style>
/* ── Global ────────────────────────────────────────────────────── */
body, .stApp {
    background-color: #0D1B2A;
    color: #E8EAF6;
    font-family: 'Segoe UI', sans-serif;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1A2744 100%);
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] * { color: #A8C4D4 !important; }
[data-testid="stSidebar"] .stRadio label { font-weight: 600; }

/* ── Banner ─────────────────────────────────────────────────────── */
.forensics-banner {
    background: linear-gradient(135deg, #0D1B2A 0%, #1A2744 50%, #0D1B2A 100%);
    border: 1px solid #1E3A5F;
    border-left: 5px solid #C0392B;
    border-radius: 8px;
    padding: 18px 28px;
    margin-bottom: 24px;
}
.banner-title {
    font-size: 22px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 1px;
    margin: 0;
}
.banner-subtitle {
    font-size: 12px;
    color: #7EC8E3;
    letter-spacing: 2px;
    margin: 4px 0 0 0;
    text-transform: uppercase;
}

/* ── KPI Metric Cards ─────────────────────────────────────────── */
.kpi-card {
    background: #1A2744;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    height: 100%;
}
.kpi-number {
    font-size: 42px;
    font-weight: 800;
    line-height: 1;
}
.kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #7EC8E3;
    margin-top: 6px;
}

/* ── Verdict Badges ─────────────────────────────────────────────── */
.badge-scam {
    background: rgba(192, 57, 43, 0.2);
    border: 2px solid #C0392B;
    border-radius: 30px;
    padding: 10px 24px;
    font-size: 20px;
    font-weight: 800;
    color: #E74C3C;
    display: inline-block;
    letter-spacing: 1px;
}
.badge-suspicious {
    background: rgba(212, 172, 13, 0.2);
    border: 2px solid #D4AC0D;
    border-radius: 30px;
    padding: 10px 24px;
    font-size: 20px;
    font-weight: 800;
    color: #F1C40F;
    display: inline-block;
    letter-spacing: 1px;
}
.badge-legitimate {
    background: rgba(30, 132, 73, 0.2);
    border: 2px solid #1E8449;
    border-radius: 30px;
    padding: 10px 24px;
    font-size: 20px;
    font-weight: 800;
    color: #27AE60;
    display: inline-block;
    letter-spacing: 1px;
}

/* ── Section Headers ─────────────────────────────────────────────── */
.section-header {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7EC8E3;
    border-bottom: 1px solid #1E3A5F;
    padding-bottom: 6px;
    margin-bottom: 16px;
}

/* ── Input widgets ────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background: #1A2744 !important;
    color: #E8EAF6 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 6px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #C0392B, #922B21) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 10px 24px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #E74C3C, #C0392B) !important;
    transform: translateY(-1px);
}

/* ── DataFrame ────────────────────────────────────────────────────── */
.stDataFrame { background: #1A2744 !important; }

/* ── Matplotlib charts ────────────────────────────────────────────── */
.chart-container {
    background: #1A2744;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 12px;
}
</style>
"""
st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────
MODEL_FILE_PATH  = "scam_model.pkl"
SCAM_THRESHOLD   = db.SCAM_THRESHOLD
SUSPICIOUS_THRESHOLD = db.SUSPICIOUS_THRESHOLD

# Keywords commonly associated with criminal scam messages
RED_FLAG_KEYWORDS = [
    "winner", "free", "prize", "claim", "urgent", "click",
    "verify", "bank", "account", "password", "loan", "cash",
    "congratulations", "selected", "reward", "exclusive", "offer",
    "guaranteed", "approved", "suspended", "alert",
]


# ─────────────────────────────────────────────────────────────────────────
#  SESSION STATE INITIALISATION
#  WHY: Streamlit re-runs the entire script on every interaction.
#       st.session_state persists variables across re-runs within one session.
# ─────────────────────────────────────────────────────────────────────────
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if "session_case_ids" not in st.session_state:
    st.session_state.session_case_ids = []

if "session_scam_count"       not in st.session_state: st.session_state.session_scam_count       = 0
if "session_suspicious_count" not in st.session_state: st.session_state.session_suspicious_count = 0
if "session_legitimate_count" not in st.session_state: st.session_state.session_legitimate_count = 0

if "last_analysis_result" not in st.session_state:
    st.session_state.last_analysis_result = None


# ─────────────────────────────────────────────────────────────────────────
#  LOAD AI MODEL
#  WHY try-except: If model_trainer.py has not been run yet, the app should
#  display a clear, actionable error rather than a raw Python traceback.
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource   # Cache: load the model only once per server session
def load_forensic_ai_model():
    """Load and cache the trained sklearn Pipeline from disk."""
    if not os.path.exists(MODEL_FILE_PATH):
        return None
    try:
        trained_pipeline = joblib.load(MODEL_FILE_PATH)
        return trained_pipeline
    except Exception as model_load_error:
        st.error(f"Model load failed: {model_load_error}")
        return None


forensic_ai_model = load_forensic_ai_model()


# ─────────────────────────────────────────────────────────────────────────
#  HELPER — Analyse a single evidence message
# ─────────────────────────────────────────────────────────────────────────
def analyse_crime_evidence(raw_evidence_text: str) -> dict:
    """
    Run the AI model on a single raw message and return a result dict.

    Steps:
        1. Clean the text using the same pipeline used during training
        2. Get class probabilities from the trained model
        3. Apply threshold logic to assign a verdict
        4. Save the result to the crime database
        5. Return all metadata for display

    Returns:
        dict: {case_id, timestamp, verdict, score, badge_html, pdf_bytes}
    """
    if forensic_ai_model is None:
        st.error("⚠️  AI Model not loaded.  Run `python model_trainer.py` first.")
        return None

    # Step 1 — Clean evidence text (same cleaning used at training time)
    cleaned_evidence = clean_crime_evidence(raw_evidence_text)

    # Step 2 — Predict class probabilities
    #   predict_proba returns [[P(ham), P(spam)]] — we want the spam probability
    probability_vector   = forensic_ai_model.predict_proba([cleaned_evidence])[0]
    scam_probability     = float(probability_vector[1])  # Index 1 = scam class

    # Step 3 — Apply threshold rules to get the human-readable verdict
    ai_verdict = db.classify_scam_verdict(scam_probability)

    # Step 4 — Persist to database
    analysis_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assigned_case_id   = db.save_case_to_database(
        original_message = raw_evidence_text,
        ai_verdict       = ai_verdict,
        confidence_score = scam_probability,
    )

    # Step 5 — Generate PDF bytes
    pdf_bytes = generate_forensic_pdf_report(
        case_id          = assigned_case_id,
        timestamp        = analysis_timestamp,
        evidence_message = raw_evidence_text,
        ai_verdict       = ai_verdict,
        confidence_score = scam_probability,
    )

    # Step 6 — Update session state counters
    st.session_state.session_case_ids.append(assigned_case_id)
    if   ai_verdict == "CRITICAL SCAM" : st.session_state.session_scam_count       += 1
    elif ai_verdict == "SUSPICIOUS"    : st.session_state.session_suspicious_count += 1
    else                               : st.session_state.session_legitimate_count += 1

    return {
        "case_id"   : assigned_case_id,
        "timestamp" : analysis_timestamp,
        "verdict"   : ai_verdict,
        "score"     : scam_probability,
        "pdf_bytes" : pdf_bytes,
    }


# ─────────────────────────────────────────────────────────────────────────
#  HELPER — Matplotlib chart: Verdict Pie Chart
# ─────────────────────────────────────────────────────────────────────────
def render_verdict_pie_chart(stats: dict) -> plt.Figure:
    """
    Create a dark-themed pie chart of verdict distribution.
    Returns a Matplotlib Figure object (displayed via st.pyplot).
    """
    labels_raw  = ["Critical Scam", "Suspicious", "Legitimate"]
    sizes_raw   = [stats["scam"], stats["suspicious"], stats["legitimate"]]
    colours_raw = ["#C0392B", "#D4AC0D", "#1E8449"]
    explode_raw = [0.06, 0.03, 0.0]

    # Filter out zero-value slices (avoids ugly "0%" wedges)
    filtered = [
        (l, s, c, e)
        for l, s, c, e in zip(labels_raw, sizes_raw, colours_raw, explode_raw)
        if s > 0
    ]
    if not filtered:
        filtered = [("No Data", 1, "#334155", 0)]

    labels, sizes, colours, explode = zip(*filtered)

    fig, chart_axis = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1A2744")
    chart_axis.set_facecolor("#1A2744")

    wedges, label_texts, auto_pcts = chart_axis.pie(
        sizes,
        labels      = labels,
        colors      = colours,
        explode     = explode,
        autopct     = lambda pct: f"{pct:.1f}%" if pct > 0 else "",
        startangle  = 140,
        textprops   = {"color": "#E8EAF6", "fontsize": 9},
        wedgeprops  = {"edgecolor": "#0D1B2A", "linewidth": 1.5},
        pctdistance = 0.75,
    )
    for auto_pct in auto_pcts:
        auto_pct.set_color("#FFFFFF")
        auto_pct.set_fontweight("bold")

    chart_axis.set_title(
        "Verdict Distribution",
        color="#7EC8E3", fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────
#  HELPER — Matplotlib chart: Red-Flag Keyword Bar Chart
# ─────────────────────────────────────────────────────────────────────────
def render_keyword_bar_chart(database_dataframe: pd.DataFrame) -> plt.Figure:
    """
    Count occurrences of each red-flag keyword in the Message column
    and render a horizontal bar chart.
    """
    fig, chart_axis = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#1A2744")
    chart_axis.set_facecolor("#1A2744")

    if database_dataframe.empty or "Message" not in database_dataframe.columns:
        chart_axis.text(
            0.5, 0.5, "No data available",
            ha="center", va="center", color="#7EC8E3", fontsize=12,
            transform=chart_axis.transAxes,
        )
        return fig

    combined_message_text = " ".join(
        database_dataframe["Message"].fillna("").astype(str)
    ).lower()

    keyword_frequency_counts = {
        keyword: combined_message_text.count(keyword)
        for keyword in RED_FLAG_KEYWORDS
    }

    # Sort descending and take top 15 for readability
    sorted_keyword_frequencies = dict(
        sorted(keyword_frequency_counts.items(), key=lambda item: item[1], reverse=True)[:15]
    )

    bars = chart_axis.barh(
        list(sorted_keyword_frequencies.keys()),
        list(sorted_keyword_frequencies.values()),
        color="#C0392B",
        edgecolor="#0D1B2A",
        linewidth=0.8,
    )

    # Colour-code: top 5 bars in crimson, rest in muted red
    for bar_index, bar in enumerate(bars):
        bar.set_color("#C0392B" if bar_index < 5 else "#922B21")

    chart_axis.set_xlabel("Occurrences", color="#A8C4D4", fontsize=9)
    chart_axis.set_title(
        "Top Red-Flag Keywords in Evidence",
        color="#7EC8E3", fontsize=11, fontweight="bold",
    )
    chart_axis.tick_params(colors="#A8C4D4", labelsize=8)
    chart_axis.spines["bottom"].set_color("#1E3A5F")
    chart_axis.spines["left"].set_color("#1E3A5F")
    chart_axis.spines["top"].set_visible(False)
    chart_axis.spines["right"].set_visible(False)
    chart_axis.invert_yaxis()

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px 0;'>
        <div style='font-size:36px;'>🔍</div>
        <div style='font-weight:800; font-size:14px; color:#FFFFFF; letter-spacing:1px;'>
            AR Forensics & CyberSecurity Labs
        </div>
        <div style='font-size:10px; color:#7EC8E3; letter-spacing:2px; margin-top:4px;'>
            SCAM DETECTION SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation menu
    selected_page = st.radio(
        "NAVIGATION",
        options=["📊  Dashboard", "🔬  New Analysis", "📁  Crime Records"],
        label_visibility="visible",
    )

    st.divider()

    # Model status indicator
    if forensic_ai_model is not None:
        st.markdown(
            "<div style='color:#27AE60; font-size:11px; font-weight:700;'>"
            "✅  AI MODEL  —  ONLINE</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#C0392B; font-size:11px; font-weight:700;'>"
            "❌  AI MODEL  —  OFFLINE<br>"
            "<span style='font-weight:400;'>Run model_trainer.py</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Session counters in the sidebar
    st.markdown("**SESSION STATS**")
    st.markdown(
        f"🔴 Scams: **{st.session_state.session_scam_count}**  "
        f"🟡 Suspicious: **{st.session_state.session_suspicious_count}**  "
        f"🟢 Legit: **{st.session_state.session_legitimate_count}**"
    )

    # Write session report button
    if st.button("📄  Export Session Report"):
        try:
            db.write_session_report({
                "session_start"    : st.session_state.session_start_time,
                "session_end"      : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_analysed"   : len(st.session_state.session_case_ids),
                "scam_count"       : st.session_state.session_scam_count,
                "suspicious_count" : st.session_state.session_suspicious_count,
                "legitimate_count" : st.session_state.session_legitimate_count,
                "case_ids"         : st.session_state.session_case_ids,
            })
            st.success("✅  session_report.txt saved!")
        except Exception as report_error:
            st.error(f"Failed: {report_error}")

    st.divider()
    st.markdown(
        "<div style='font-size:9px; color:#4A6887; text-align:center;'>"
        "© 2026 AR Forensics & CyberSecurity Labs</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
#  BANNER — displayed on every page
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='forensics-banner'>
    <p class='banner-title'>🔬 AR Forensics &amp; CyberSecurity Labs — Spam Detection System</p>
    <p class='banner-subtitle'>AI-Powered Scam Detection &amp; Case Management Platform
    &nbsp;|&nbsp; AR Forensics &amp; CyberSecurity Labs</p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════
if "Dashboard" in selected_page:

    st.markdown("<div class='section-header'>LIVE CASE STATISTICS</div>", unsafe_allow_html=True)

    # Fetch current database stats
    live_statistics = db.compute_database_statistics()

    # ── KPI Cards ─────────────────────────────────────────────────────────
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-number' style='color:#7EC8E3;'>{live_statistics['total']}</div>
            <div class='kpi-label'>Total Cases</div>
        </div>""", unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-number' style='color:#E74C3C;'>{live_statistics['scam']}</div>
            <div class='kpi-label'>🔴 Critical Scams</div>
        </div>""", unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-number' style='color:#F1C40F;'>{live_statistics['suspicious']}</div>
            <div class='kpi-label'>🟡 Suspicious</div>
        </div>""", unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-number' style='color:#27AE60;'>{live_statistics['legitimate']}</div>
            <div class='kpi-label'>🟢 Legitimate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>VISUAL ANALYTICS</div>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([1, 1.4])

    with chart_col1:
        pie_figure = render_verdict_pie_chart(live_statistics)
        st.pyplot(pie_figure, use_container_width=True)
        plt.close(pie_figure)   # Free memory

    with chart_col2:
        all_cases_for_chart = db.load_all_case_records()
        bar_figure = render_keyword_bar_chart(all_cases_for_chart)
        st.pyplot(bar_figure, use_container_width=True)
        plt.close(bar_figure)

    # ── Recent 5 cases ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>RECENT CASES</div>", unsafe_allow_html=True)
    recent_cases = db.load_all_case_records()

    if recent_cases.empty:
        st.info("No cases recorded yet.  Use **New Analysis** to submit evidence.")
    else:
        display_cols = ["Case_ID", "Timestamp", "Verdict", "Score"]
        st.dataframe(
            recent_cases[display_cols].tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 2 — NEW ANALYSIS
# ═════════════════════════════════════════════════════════════════════════
elif "New Analysis" in selected_page:

    st.markdown("<div class='section-header'>SUBMIT EVIDENCE FOR FORENSIC ANALYSIS</div>",
                unsafe_allow_html=True)

    # Provide example evidence texts for quick testing (viva demo)
    example_evidence_texts = {
        "-- Select an example --"      : "",
        "🔴 Example: Prize Scam"       : "WINNER!! Congratulations! You have been selected to claim a FREE prize of Rs.50,000. Urgent: Call now to verify your bank account and claim before offer expires!",
        "🟡 Example: Phishing Attempt" : "Your account has been flagged. Please verify your details by clicking the link provided, or your access will be restricted.",
        "🟢 Example: Normal Message"   : "Hey, are you coming to the Digital Forensics lecture tomorrow morning? Let me know so we can meet at the library.",
    }
    selected_example = st.selectbox("Quick Evidence Examples:", list(example_evidence_texts.keys()))

    # Pre-fill text area if an example was selected
    example_prefill = example_evidence_texts[selected_example]

    # Evidence text input area
    crime_evidence_text = st.text_area(
        "📋  Evidence Text (paste the suspicious message here):",
        value       = example_prefill,
        height      = 160,
        placeholder = "Enter the scam / spam message text to analyse …",
        help        = "Paste the full text of the suspicious message.  The AI will classify it.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyse_button_pressed = st.button("🔬  ANALYSE EVIDENCE", use_container_width=False)

    # ── Run Analysis ──────────────────────────────────────────────────────
    if analyse_button_pressed:
        if not crime_evidence_text.strip():
            st.warning("⚠️  Please enter some evidence text before running analysis.")
        elif forensic_ai_model is None:
            st.error(
                "❌  AI Model not loaded.  "
                "Run `python model_trainer.py` in the terminal first."
            )
        else:
            with st.spinner("🔬  Analysing evidence …  Please wait."):
                analysis_result = analyse_crime_evidence(crime_evidence_text)

            if analysis_result:
                st.session_state.last_analysis_result = analysis_result
                st.session_state.last_evidence_text   = crime_evidence_text

    # ── Display Result ────────────────────────────────────────────────────
    if st.session_state.last_analysis_result:
        result       = st.session_state.last_analysis_result
        evidence_txt = st.session_state.get("last_evidence_text", "")

        st.divider()
        st.markdown("<div class='section-header'>ANALYSIS RESULT</div>", unsafe_allow_html=True)

        # ── Verdict Badge ──────────────────────────────────────────────────
        verdict_str = result["verdict"]
        if verdict_str == "CRITICAL SCAM":
            badge_html = f"<div class='badge-scam'>🔴 CRITICAL SCAM</div>"
        elif verdict_str == "SUSPICIOUS":
            badge_html = f"<div class='badge-suspicious'>🟡 SUSPICIOUS</div>"
        else:
            badge_html = f"<div class='badge-legitimate'>🟢 LEGITIMATE</div>"

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Case metadata
            st.markdown(f"**Case ID:** `{result['case_id']}`")
            st.markdown(f"**Timestamp:** {result['timestamp']}")
            st.markdown(f"**Confidence Score:** {result['score'] * 100:.2f}%")

        with res_col2:
            # Confidence gauge (simple progress bar visual)
            progress_colour = (
                "#C0392B" if result["score"] >= SCAM_THRESHOLD
                else "#D4AC0D" if result["score"] >= SUSPICIOUS_THRESHOLD
                else "#1E8449"
            )
            bar_pct  = int(result["score"] * 100)
            st.markdown(
                f"<div style='font-size:11px; color:#7EC8E3; font-weight:700; "
                f"letter-spacing:1px; margin-bottom:6px;'>AI CONFIDENCE GAUGE</div>"
                f"<div style='background:#0D1B2A; border-radius:20px; height:22px; "
                f"border:1px solid #1E3A5F;'>"
                f"<div style='background:{progress_colour}; width:{bar_pct}%; "
                f"height:100%; border-radius:20px; display:flex; align-items:center; "
                f"padding-left:10px; font-size:11px; font-weight:800; color:white;'>"
                f"{bar_pct}%</div></div>",
                unsafe_allow_html=True,
            )

        # Threshold reference
        st.caption(
            f"Thresholds: 🔴 ≥ {int(SCAM_THRESHOLD*100)}%  "
            f"🟡 {int(SUSPICIOUS_THRESHOLD*100)}–{int(SCAM_THRESHOLD*100)-1}%  "
            f"🟢 < {int(SUSPICIOUS_THRESHOLD*100)}%"
        )

        # ── PDF Download ───────────────────────────────────────────────────
        st.divider()
        st.markdown("<div class='section-header'>FORENSIC REPORT</div>", unsafe_allow_html=True)

        pdf_filename = f"ForensicReport_{result['case_id']}.pdf"
        st.download_button(
            label     = "📄  Download Official Forensic PDF Report",
            data      = result["pdf_bytes"],
            file_name = pdf_filename,
            mime      = "application/pdf",
            help      = "Download the court-ready PDF forensic report for this case.",
        )
        st.caption(f"Report file: `{pdf_filename}`  |  Case ID: `{result['case_id']}`")


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CRIME RECORDS
# ═════════════════════════════════════════════════════════════════════════
elif "Crime Records" in selected_page:

    st.markdown("<div class='section-header'>CRIME DATABASE — CASE RECORDS</div>",
                unsafe_allow_html=True)

    # ── Search Controls ────────────────────────────────────────────────────
    search_col1, search_col2 = st.columns([3, 1])

    with search_col1:
        search_keyword_input = st.text_input(
            "🔍  Search by keyword in evidence message:",
            placeholder = "e.g. winner, bank, click, congratulations …",
        )
    with search_col2:
        case_id_search_input = st.text_input(
            "🗂️  Search by Case ID:",
            placeholder = "AR-XXXXXXXX",
        )

    # ── Verdict filter ────────────────────────────────────────────────────
    verdict_filter_options = ["All Verdicts", "CRITICAL SCAM", "SUSPICIOUS", "LEGITIMATE"]
    selected_verdict_filter = st.selectbox("Filter by Verdict:", verdict_filter_options)

    # ── Load and filter data ───────────────────────────────────────────────
    if case_id_search_input.strip():
        filtered_cases_dataframe = db.fetch_case_by_id(case_id_search_input.strip())
        filter_description = f"Case ID: `{case_id_search_input.strip()}`"
    elif search_keyword_input.strip():
        filtered_cases_dataframe = db.search_case_records_by_keyword(search_keyword_input.strip())
        filter_description = f"Keyword: `{search_keyword_input.strip()}`"
    else:
        filtered_cases_dataframe = db.load_all_case_records()
        filter_description = "All records"

    # Apply verdict filter
    if selected_verdict_filter != "All Verdicts" and not filtered_cases_dataframe.empty:
        filtered_cases_dataframe = filtered_cases_dataframe[
            filtered_cases_dataframe["Verdict"] == selected_verdict_filter
        ]

    # ── Results display ────────────────────────────────────────────────────
    records_count = len(filtered_cases_dataframe)
    st.markdown(f"**Found:** {records_count} record(s)  —  {filter_description}")

    if filtered_cases_dataframe.empty:
        st.info("No matching records found in the crime database.")
    else:
        # Colour-code the Verdict column via pandas Styler
        def apply_verdict_row_style(row):
            """Apply background colour to entire row based on Verdict."""
            colour_map = {
                "CRITICAL SCAM" : "background-color: rgba(192, 57, 43, 0.12); color: #E74C3C;",
                "SUSPICIOUS"    : "background-color: rgba(212, 172, 13, 0.12); color: #F1C40F;",
                "LEGITIMATE"    : "background-color: rgba(30, 132, 73, 0.12);  color: #27AE60;",
            }
            verdict_cell_style = colour_map.get(row.get("Verdict", ""), "")
            return [verdict_cell_style if col == "Verdict" else "" for col in row.index]

        styled_dataframe = filtered_cases_dataframe.style.apply(apply_verdict_row_style, axis=1)

        st.dataframe(styled_dataframe, use_container_width=True, hide_index=True)

        # ── CSV Export ──────────────────────────────────────────────────────
        csv_export_bytes = filtered_cases_dataframe.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇️  Export Filtered Results as CSV",
            data      = csv_export_bytes,
            file_name = f"crime_records_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime      = "text/csv",
        )

    # ── Full record detail expander ────────────────────────────────────────
    st.divider()
    st.markdown("<div class='section-header'>RECORD DETAIL VIEWER</div>", unsafe_allow_html=True)

    all_case_ids = db.load_all_case_records().get("Case_ID", pd.Series()).tolist()

    if all_case_ids:
        selected_case_id_detail = st.selectbox("Select Case ID to inspect:", ["--"] + all_case_ids)

        if selected_case_id_detail != "--":
            case_detail_df = db.fetch_case_by_id(selected_case_id_detail)
            if not case_detail_df.empty:
                detail_row = case_detail_df.iloc[0]

                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.markdown(f"**Case ID:** `{detail_row['Case_ID']}`")
                    st.markdown(f"**Timestamp:** {detail_row['Timestamp']}")
                    st.markdown(f"**Verdict:** {detail_row['Verdict']}")
                    st.markdown(f"**Score:** {float(detail_row['Score'])*100:.2f}%")
                with detail_col2:
                    st.markdown("**Evidence Text:**")
                    st.text_area("", value=detail_row["Message"], height=100, disabled=True, label_visibility="collapsed")

                # Re-generate PDF for this historical case
                try:
                    historical_pdf_bytes = generate_forensic_pdf_report(
                        case_id          = detail_row["Case_ID"],
                        timestamp        = detail_row["Timestamp"],
                        evidence_message = detail_row["Message"],
                        ai_verdict       = detail_row["Verdict"],
                        confidence_score = float(detail_row["Score"]),
                    )
                    st.download_button(
                        label     = f"📄  Download PDF Report — {detail_row['Case_ID']}",
                        data      = historical_pdf_bytes,
                        file_name = f"ForensicReport_{detail_row['Case_ID']}.pdf",
                        mime      = "application/pdf",
                    )
                except Exception as pdf_error:
                    st.warning(f"Could not generate PDF: {pdf_error}")
    else:
        st.info("No cases in database yet.  Submit evidence in **New Analysis** first.")
