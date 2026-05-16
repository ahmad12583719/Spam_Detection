"""
=============================================================================
  pdf_report_generator.py  
  AI-Driven Criminal Scam Analysis & Case Tracking System
=============================================================================
  PURPOSE  : Generate a formal, court-ready Forensic PDF Report for a single
             case using the ReportLab library.

  FORMAT   : A4 page, black-and-white professional layout with:
               • Official letterhead / header
               • Case metadata table
               • Evidence section
               • AI verdict panel
               • Digital signature block

  AUTHOR   : Ahmad Raza 
=============================================================================
"""

# ── Standard-library ──────────────────────────────────────────────────────
import io          # In-memory byte buffer — avoids writing temp files to disk
import datetime    # For stamping the report generation time

# ── ReportLab ─────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes    import A4
from reportlab.lib.units        import cm
from reportlab.lib              import colors
from reportlab.lib.styles       import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums        import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus         import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable,
)

# ─────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE — consistent with the Streamlit UI's badge colours
# ─────────────────────────────────────────────────────────────────────────
COLOUR_DEEP_NAVY   = colors.HexColor("#0D1B2A")   # Primary header background
COLOUR_CRIMSON     = colors.HexColor("#C0392B")   # Scam verdict highlight
COLOUR_AMBER       = colors.HexColor("#D4AC0D")   # Suspicious verdict highlight
COLOUR_FOREST      = colors.HexColor("#1E8449")   # Legitimate verdict highlight
COLOUR_LIGHT_GREY  = colors.HexColor("#F2F3F4")   # Alternating table row fill
COLOUR_WHITE       = colors.white


def _get_verdict_colour(ai_verdict: str) -> colors.Color:
    """Return the appropriate ReportLab colour for a given verdict string."""
    verdict_upper = ai_verdict.upper()
    if "SCAM"        in verdict_upper: return COLOUR_CRIMSON
    if "SUSPICIOUS"  in verdict_upper: return COLOUR_AMBER
    return COLOUR_FOREST


def _get_verdict_emoji(ai_verdict: str) -> str:
    """Return status badge emoji matching the verdict."""
    verdict_upper = ai_verdict.upper()
    if "SCAM"       in verdict_upper: return "🔴 CRITICAL SCAM"
    if "SUSPICIOUS" in verdict_upper: return "🟡 SUSPICIOUS"
    return "🟢 LEGITIMATE"


# ─────────────────────────────────────────────────────────────────────────
#  MAIN REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────
def generate_forensic_pdf_report(
    case_id          : str,
    timestamp        : str,
    evidence_message : str,
    ai_verdict       : str,
    confidence_score : float,
) -> bytes:
    """
    Render a formal Forensic PDF Report and return it as raw bytes.

    The caller (app.py) feeds these bytes directly into Streamlit's
    st.download_button — no temporary file on disk is required.

    Args:
        case_id          (str)  : Unique case identifier  (e.g. 'AR-A3F1B2C4')
        timestamp        (str)  : When the analysis was performed
        evidence_message (str)  : Original text submitted for analysis
        ai_verdict       (str)  : 'CRITICAL SCAM' | 'SUSPICIOUS' | 'LEGITIMATE'
        confidence_score (float): AI probability score (0.0 – 1.0)

    Returns:
        bytes: Complete PDF binary content, ready to serve or write to disk.
    """

    # ── Buffer setup — write PDF bytes to memory, not a file ─────────────
    pdf_byte_buffer = io.BytesIO()
    _u = case_id.replace("-", "")

    # ── Document setup ────────────────────────────────────────────────────
    forensic_report_doc = SimpleDocTemplate(
        pdf_byte_buffer,
        pagesize    = A4,
        topMargin   = 1.8 * cm,
        bottomMargin= 2.0 * cm,
        leftMargin  = 2.0 * cm,
        rightMargin = 2.0 * cm,
    )

    # ── Styles ────────────────────────────────────────────────────────────
    base_styles = getSampleStyleSheet()

    style_header_main = ParagraphStyle(
        f"HeaderMain_{_u }",  # Unique style name to avoid conflicts
        parent    = base_styles["Heading1"],
        fontSize  = 16,
        textColor = COLOUR_WHITE,
        alignment = TA_CENTER,
        spaceAfter= 2,
        fontName  = "Helvetica-Bold",
    )
    style_header_sub = ParagraphStyle(
        f"HeaderSub_{_u }",
        parent    = base_styles["Normal"],
        fontSize  = 9,
        textColor = COLOUR_LIGHT_GREY,
        alignment = TA_CENTER,
        fontName  = "Helvetica",
    )
    style_section_title = ParagraphStyle(
        f"SectionTitle_{_u }",
        parent    = base_styles["Heading2"],
        fontSize  = 11,
        textColor = COLOUR_DEEP_NAVY,
        fontName  = "Helvetica-Bold",
        spaceBefore=10,
        spaceAfter = 4,
    )
    style_body = ParagraphStyle(
        f"Body_{_u }",
        parent    = base_styles["Normal"],
        fontSize  = 9,
        textColor = colors.black,
        leading   = 14,
        alignment = TA_JUSTIFY,
        fontName  = "Helvetica",
    )
    style_evidence_box = ParagraphStyle(
        f"EvidenceBox_{_u }",
        parent     = base_styles["Normal"],
        fontSize   = 9,
        textColor  = colors.black,
        fontName   = "Courier",
        leading    = 14,
        backColor  = COLOUR_LIGHT_GREY,
        leftIndent = 6,
        rightIndent= 6,
        borderPad  = 4,
    )
    style_verdict_text = ParagraphStyle(
        f"VerdictText_{_u }",
        parent    = base_styles["Normal"],
        fontSize  = 14,
        textColor = _get_verdict_colour(ai_verdict),
        fontName  = "Helvetica-Bold",
        alignment = TA_CENTER,
    )
    style_footer = ParagraphStyle(
        f"Footer_{_u }",
        parent    = base_styles["Normal"],
        fontSize  = 7.5,
        textColor = colors.grey,
        alignment = TA_CENTER,
        fontName  = "Helvetica-Oblique",
    )
    style_disclaimer = ParagraphStyle(
        f"Disclaimer_{_u }",
        parent    = base_styles["Normal"],
        fontSize  = 8,
        textColor = colors.HexColor("#555555"),
        alignment = TA_CENTER,
        fontName  = "Helvetica-Oblique",
        spaceBefore=6,
    )

    # ── Story (content elements in order) ─────────────────────────────────
    story_elements = []

    # ── 1. OFFICIAL LETTERHEAD ────────────────────────────────────────────
    letterhead_data = [
        [Paragraph("Criminal Scam Analysis Lab", style_header_main)],
        [Paragraph("Digital Forensics & Cyber Security AI Tool", style_header_sub)],
        [Paragraph("AR Forensics & CyberSecurity Labs, Lahore, Pakistan", style_header_sub)],
        [Paragraph("— OFFICIAL FORENSIC CASE REPORT —", style_header_sub)],
    ]
    letterhead_table = Table(letterhead_data, colWidths=["100%"])
    letterhead_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), COLOUR_DEEP_NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS",(0, 0), (-1, -1), [4, 4, 4, 4]),
    ]))
    story_elements.append(letterhead_table)
    story_elements.append(Spacer(1, 0.4 * cm))

    # ── 2. CASE METADATA TABLE ────────────────────────────────────────────
    story_elements.append(Paragraph("CASE IDENTIFICATION", style_section_title))
    story_elements.append(HRFlowable(width="100%", thickness=1, color=COLOUR_DEEP_NAVY))
    story_elements.append(Spacer(1, 0.2 * cm))

    report_generated_at = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    metadata_rows = [
        ["Case ID",              case_id],
        ["Analysis Timestamp",   timestamp],
        ["Report Generated",     report_generated_at],
        ["Analyst System",       "Criminal Scam Analysis System v1.0"],
        ["AI Engine",            "Multinomial Naive Bayes + TF-IDF (sklearn)"],
        ["Classification Model", "spam_model.pkl (Trained on SMS Spam Dataset)"],
    ]

    cell_style_label = ParagraphStyle(
        f"CellLabel_{_u }", parent=base_styles["Normal"],
        fontSize=8.5, fontName="Helvetica-Bold", textColor=COLOUR_DEEP_NAVY,
    )
    cell_style_value = ParagraphStyle(
        f"CellValue_{_u }", parent=base_styles["Normal"],
        fontSize=8.5, fontName="Helvetica", textColor=colors.black,
    )

    metadata_table_data = [
        [
            Paragraph(label, cell_style_label),
            Paragraph(value, cell_style_value),
        ]
        for label, value in metadata_rows
    ]

    metadata_table = Table(
        metadata_table_data,
        colWidths=[5 * cm, 11.5 * cm],
    )
    metadata_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), COLOUR_LIGHT_GREY),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [COLOUR_WHITE, COLOUR_LIGHT_GREY]),
    ]))

    story_elements.append(metadata_table)
    story_elements.append(Spacer(1, 0.4 * cm))

    # ── 3. EVIDENCE SECTION ───────────────────────────────────────────────
    story_elements.append(Paragraph("EVIDENCE TEXT (Exhibit A)", style_section_title))
    story_elements.append(HRFlowable(width="100%", thickness=1, color=COLOUR_DEEP_NAVY))
    story_elements.append(Spacer(1, 0.2 * cm))

    # Escape HTML special characters so ReportLab does not misparse them
    safe_evidence_text = (
        evidence_message
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    story_elements.append(Paragraph(safe_evidence_text, style_evidence_box))
    story_elements.append(Spacer(1, 0.1 * cm))
    story_elements.append(
        Paragraph(
            f"Character count: {len(evidence_message)}  |  Word count: {len(evidence_message.split())}",
            style_disclaimer,
        )
    )
    story_elements.append(Spacer(1, 0.4 * cm))

    # ── 4. AI VERDICT PANEL ───────────────────────────────────────────────
    story_elements.append(Paragraph("AI FORENSIC VERDICT", style_section_title))
    story_elements.append(HRFlowable(width="100%", thickness=1, color=COLOUR_DEEP_NAVY))
    story_elements.append(Spacer(1, 0.2 * cm))

    verdict_colour    = _get_verdict_colour(ai_verdict)
    verdict_badge_str = _get_verdict_emoji(ai_verdict)

    verdict_panel_data = [
        [
            Paragraph(verdict_badge_str, style_verdict_text),
        ],
        [
            Paragraph(
                f"AI Confidence Score: <b>{confidence_score * 100:.2f}%</b>",
                ParagraphStyle(f"Conf_{_u }", parent=base_styles["Normal"],
                               fontSize=10, alignment=TA_CENTER, fontName="Helvetica"),
            ),
        ],
    ]
    verdict_table = Table(verdict_panel_data, colWidths=["100%"])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.Color(
            verdict_colour.red, verdict_colour.green, verdict_colour.blue, alpha=0.08
        )),
        ("BOX",           (0, 0), (-1, -1), 2, verdict_colour),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story_elements.append(verdict_table)
    story_elements.append(Spacer(1, 0.4 * cm))

    # ── 5. INTERPRETATION NOTES ───────────────────────────────────────────
    story_elements.append(Paragraph("ANALYST INTERPRETATION NOTES", style_section_title))
    story_elements.append(HRFlowable(width="100%", thickness=1, color=COLOUR_DEEP_NAVY))
    story_elements.append(Spacer(1, 0.2 * cm))

    interpretation_map = {
        "CRITICAL SCAM": (
            "The submitted evidence text has been classified as a HIGH-RISK criminal scam "
            "message.  The AI model assigned a confidence score exceeding the critical "
            "threshold (≥70%), indicating strong linguistic patterns associated with "
            "fraudulent communications, including urgency manipulation, unsolicited "
            "prize/reward claims, and requests for personal or financial information. "
            "Immediate escalation to a senior forensic investigator is recommended."
        ),
        "SUSPICIOUS": (
            "The submitted evidence text has been flagged as SUSPICIOUS.  The AI model "
            "detected moderate scam-associated linguistic patterns (confidence: 40–70%). "
            "This message warrants further manual review.  Cross-reference with the crime "
            "database for similar reported patterns before issuing a final determination."
        ),
        "LEGITIMATE": (
            "The submitted evidence text has been classified as LEGITIMATE.  The AI model "
            "found no significant indicators of fraudulent intent (confidence score below "
            "the 40% suspicious threshold).  This message exhibits language patterns "
            "consistent with normal, benign communication.  No immediate escalation required."
        ),
    }
    interpretation_key = (
        "CRITICAL SCAM" if "SCAM" in ai_verdict.upper()
        else "SUSPICIOUS" if "SUSPICIOUS" in ai_verdict.upper()
        else "LEGITIMATE"
    )
    story_elements.append(
        Paragraph(interpretation_map[interpretation_key], style_body)
    )
    story_elements.append(Spacer(1, 0.5 * cm))

    # ── 6. DIGITAL SIGNATURE BLOCK ────────────────────────────────────────
    story_elements.append(Paragraph("AUTHORISATION & SIGNATURE", style_section_title))
    story_elements.append(HRFlowable(width="100%", thickness=1, color=COLOUR_DEEP_NAVY))
    story_elements.append(Spacer(1, 0.3 * cm))

    signature_rows = [
        ["Lead Forensic Analyst:", "______________________________", "Badge ID:", "________"],
        ["Supervisor Approval:",   "______________________________", "Date:",     "________"],
    ]
    sig_style = ParagraphStyle(
        f"Sig_{_u }", parent=base_styles["Normal"],
        fontSize=8, fontName="Helvetica-Bold",
    )
    sig_value_style = ParagraphStyle(
        f"SigVal_{_u }", parent=base_styles["Normal"],
        fontSize=8, fontName="Helvetica",
    )
    sig_table_data = [
        [
            Paragraph(row[0], sig_style), Paragraph(row[1], sig_value_style),
            Paragraph(row[2], sig_style), Paragraph(row[3], sig_value_style),
        ]
        for row in signature_rows
    ]
    sig_table = Table(sig_table_data, colWidths=[4 * cm, 5.5 * cm, 2.5 * cm, 4.5 * cm])
    sig_table.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
    ]))
    story_elements.append(sig_table)
    story_elements.append(Spacer(1, 0.5 * cm))

    # ── 7. FOOTER ─────────────────────────────────────────────────────────
    story_elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story_elements.append(Spacer(1, 0.15 * cm))
    story_elements.append(
        Paragraph(
            f"Case {case_id}  |  CONFIDENTIAL — For official forensic use only  "
            f"|  AR Forensics & CyberSecurity Labs  |  {report_generated_at}",
            style_footer,
        )
    )

    # ── Build (render) the PDF ─────────────────────────────────────────────
    forensic_report_doc.build(story_elements)

    # Return raw bytes from the in-memory buffer
    pdf_bytes = pdf_byte_buffer.getvalue()
    pdf_byte_buffer.close()
    return pdf_bytes
