import warnings
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import shap
import io
import sqlite3
import hashlib
import json
from datetime import datetime
from sklearn.exceptions import InconsistentVersionWarning
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from streamlit_option_menu import option_menu
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .ri-normal  { background:rgba(34,197,94,0.15);  border-left:3px solid #22C55E; border-radius:6px; padding:8px 14px; margin:4px 0; color:#86EFAC; font-size:0.85rem; font-weight:500; }
    .ri-warning { background:rgba(234,179,8,0.15);  border-left:3px solid #EAB308; border-radius:6px; padding:8px 14px; margin:4px 0; color:#FDE047; font-size:0.85rem; font-weight:500; }
    .ri-danger  { background:rgba(239,68,68,0.15);  border-left:3px solid #EF4444; border-radius:6px; padding:8px 14px; margin:4px 0; color:#FCA5A5; font-size:0.85rem; font-weight:500; }
    .result-positive { background:rgba(239,68,68,0.1);  border:1px solid rgba(239,68,68,0.4);  border-radius:12px; padding:20px 24px; margin:12px 0; }
    .result-negative { background:rgba(34,197,94,0.1);  border:1px solid rgba(34,197,94,0.4);  border-radius:12px; padding:20px 24px; margin:12px 0; }
    .result-title-pos { font-size:1.2rem; font-weight:700; color:#FCA5A5; margin-bottom:6px; }
    .result-title-neg { font-size:1.2rem; font-weight:700; color:#86EFAC; margin-bottom:6px; }
    .result-sub { font-size:0.84rem; color:#94A3B8; margin-top:8px; line-height:1.5; }
    .rec-card        { background:rgba(59,130,246,0.1);  border-left:3px solid #3B82F6; border-radius:6px; padding:10px 14px; margin:5px 0; color:#CBD5E1; font-size:0.87rem; line-height:1.5; }
    .rec-card-urgent { background:rgba(249,115,22,0.1);  border-left:3px solid #F97316; color:#FED7AA; }
    .disease-card { background:rgba(30,41,59,0.8); border:1px solid rgba(59,130,246,0.3); border-radius:12px; padding:24px; margin-top:8px; }
    .disease-name { font-size:1.6rem; font-weight:700; color:#F1F5F9; margin:8px 0; }
    .disease-label-pos { background:rgba(239,68,68,0.2); color:#FCA5A5; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; display:inline-block; }
    .disease-label-med { background:rgba(234,179,8,0.2);  color:#FDE047; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; display:inline-block; }
    .disease-label-neg { background:rgba(34,197,94,0.2);  color:#86EFAC; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; display:inline-block; }
    .ins-flag-high   { background:rgba(249,115,22,0.1); border-left:3px solid #F97316; border-radius:6px; padding:8px 14px; margin:4px 0; color:#FED7AA; font-size:0.84rem; }
    .ins-flag-medium { background:rgba(234,179,8,0.1);  border-left:3px solid #EAB308; border-radius:6px; padding:8px 14px; margin:4px 0; color:#FDE047; font-size:0.84rem; }
    .ins-flag-low    { background:rgba(34,197,94,0.1);  border-left:3px solid #22C55E; border-radius:6px; padding:8px 14px; margin:4px 0; color:#86EFAC; font-size:0.84rem; }
    .ins-flag-critical { background:rgba(239,68,68,0.15); border-left:3px solid #EF4444; border-radius:6px; padding:8px 14px; margin:4px 0; color:#FCA5A5; font-size:0.84rem; font-weight:600; }
    .ins-metric { background:rgba(30,41,59,0.8); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:16px; text-align:center; }
    .ins-metric-label { font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }
    .ins-metric-value { font-size:1.5rem; font-weight:700; }
    .patient-card { background:rgba(15,23,42,0.8); border:1px solid rgba(59,130,246,0.25); border-radius:12px; padding:20px 24px; margin-bottom:16px; }
    .patient-card-selected { background:rgba(29,78,216,0.15); border:1px solid rgba(59,130,246,0.5); border-radius:12px; padding:20px 24px; margin-bottom:16px; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════

DB_PATH = "medipredict.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    # Employees (insurance company staff)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            email TEXT,
            employee_id TEXT,
            department TEXT,
            company TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    # Patients registered by employees
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_by_user_id INTEGER NOT NULL,
            patient_name TEXT NOT NULL,
            patient_id TEXT UNIQUE NOT NULL,
            phone TEXT,
            email TEXT,
            date_of_birth TEXT,
            gender TEXT,
            address TEXT,
            city TEXT,
            pincode TEXT,
            policy_number TEXT,
            nominee_name TEXT,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (created_by_user_id) REFERENCES users(id)
        )
    """)
    # Disease screenings linked to patient
    c.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            patient_id INTEGER,
            disease TEXT NOT NULL,
            result TEXT NOT NULL,
            risk_pct REAL NOT NULL,
            inputs_json TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)
    # Insurance reports linked to patient
    c.execute("""
        CREATE TABLE IF NOT EXISTS insurance_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            patient_id INTEGER,
            decision TEXT NOT NULL,
            category TEXT NOT NULL,
            risk_points INTEGER NOT NULL,
            premium_monthly REAL,
            loading_factor REAL,
            patient_info_json TEXT,
            flags_json TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)
    conn.commit()

    # ── Migrations: safely add missing columns to existing DBs ──
    migrations = [
        ("users",            "employee_id",  "ALTER TABLE users ADD COLUMN employee_id TEXT"),
        ("users",            "department",   "ALTER TABLE users ADD COLUMN department TEXT"),
        ("users",            "company",      "ALTER TABLE users ADD COLUMN company TEXT"),
        ("screenings",       "patient_id",   "ALTER TABLE screenings ADD COLUMN patient_id INTEGER"),
        ("insurance_reports","patient_id",   "ALTER TABLE insurance_reports ADD COLUMN patient_id INTEGER"),
    ]
    for table, column, sql in migrations:
        existing = [row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in existing:
            c.execute(sql)
    conn.commit()
    conn.close()

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def create_user(username, password, full_name, email, employee_id, department, company):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username,password_hash,full_name,email,employee_id,department,company) VALUES (?,?,?,?,?,?,?)",
            (username, hash_password(password), full_name, email, employee_id, department, company)
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username, hash_password(password))
    ).fetchone()
    conn.close()
    return dict(row) if row else None

# ── Patient CRUD ──────────────────────────────────────

def create_patient(user_id, name, patient_id, phone, email, dob, gender,
                   address, city, pincode, policy_number, nominee_name, notes):
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO patients
            (created_by_user_id, patient_name, patient_id, phone, email,
             date_of_birth, gender, address, city, pincode,
             policy_number, nominee_name, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (user_id, name, patient_id, phone, email, dob, gender,
              address, city, pincode, policy_number, nominee_name, notes))
        conn.commit()
        return True, "Patient registered successfully!"
    except sqlite3.IntegrityError:
        return False, "Patient ID already exists. Please use a unique Patient ID."
    finally:
        conn.close()

def get_patients(user_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM patients WHERE created_by_user_id=? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_patient_by_id(pid):
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    conn.close()
    return dict(row) if row else None

def update_patient(pid, name, phone, email, dob, gender,
                   address, city, pincode, policy_number, nominee_name, notes):
    conn = get_db()
    conn.execute("""
        UPDATE patients SET patient_name=?,phone=?,email=?,date_of_birth=?,gender=?,
        address=?,city=?,pincode=?,policy_number=?,nominee_name=?,notes=?
        WHERE id=?
    """, (name, phone, email, dob, gender, address, city, pincode,
          policy_number, nominee_name, notes, pid))
    conn.commit()
    conn.close()

def delete_patient(pid, user_id):
    conn = get_db()
    conn.execute("DELETE FROM patients WHERE id=? AND created_by_user_id=?", (pid, user_id))
    conn.execute("DELETE FROM screenings WHERE patient_id=?", (pid,))
    conn.execute("DELETE FROM insurance_reports WHERE patient_id=?", (pid,))
    conn.commit()
    conn.close()

# ── Screenings ──────────────────────────────────────

def save_screening_db(user_id, patient_id, disease, result, risk_pct, inputs):
    conn = get_db()
    conn.execute(
        "INSERT INTO screenings (user_id, patient_id, disease, result, risk_pct, inputs_json) VALUES (?,?,?,?,?,?)",
        (user_id, patient_id, disease, result, risk_pct, json.dumps(inputs))
    )
    conn.commit()
    conn.close()

def get_screenings_db(user_id, patient_id=None):
    conn = get_db()
    base_select = (
        "SELECT s.id, s.user_id, s.patient_id, s.disease, s.result, s.risk_pct, "
        "s.inputs_json, s.created_at, "
        "p.patient_name, p.patient_id as pat_code "
        "FROM screenings s LEFT JOIN patients p ON s.patient_id=p.id "
    )
    if patient_id:
        rows = conn.execute(
            base_select + "WHERE s.user_id=? AND s.patient_id=? ORDER BY s.created_at DESC",
            (user_id, patient_id)
        ).fetchall()
    else:
        rows = conn.execute(
            base_select + "WHERE s.user_id=? ORDER BY s.created_at DESC",
            (user_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_insurance_report_db(user_id, patient_id, result_data, patient_info):
    conn = get_db()
    conn.execute("""
        INSERT INTO insurance_reports
        (user_id, patient_id, decision, category, risk_points, premium_monthly,
         loading_factor, patient_info_json, flags_json)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (user_id, patient_id, result_data["decision"], result_data["category"],
          result_data["riskPoints"], result_data.get("finalPremium"),
          result_data.get("premiumMultiplier"),
          json.dumps(patient_info), json.dumps(result_data["flags"])))
    conn.commit()
    conn.close()

def get_insurance_reports_db(user_id, patient_id=None):
    conn = get_db()
    base_select = (
        "SELECT r.id, r.user_id, r.patient_id, r.decision, r.category, r.risk_points, "
        "r.premium_monthly, r.loading_factor, r.patient_info_json, r.flags_json, r.created_at, "
        "p.patient_name, p.patient_id as pat_code "
        "FROM insurance_reports r LEFT JOIN patients p ON r.patient_id=p.id "
    )
    if patient_id:
        rows = conn.execute(
            base_select + "WHERE r.user_id=? AND r.patient_id=? ORDER BY r.created_at DESC",
            (user_id, patient_id)
        ).fetchall()
    else:
        rows = conn.execute(
            base_select + "WHERE r.user_id=? ORDER BY r.created_at DESC",
            (user_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_screening_db(sid, user_id):
    conn = get_db()
    conn.execute("DELETE FROM screenings WHERE id=? AND user_id=?", (sid, user_id))
    conn.commit()
    conn.close()

init_db()

# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════

for key, default in [
    ("logged_in", False),
    ("user", None),
    ("ai_risk_scores", {"diabetes":0.0,"heart":0.0,"liver":0.0}),
    ("active_patient", None),   # currently selected patient dict
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════

def show_auth_page():
    st.markdown("""
    <div style="text-align:center;padding:30px 0 10px 0;">
        <div style="font-size:3rem;">🏥</div>
        <h1 style="color:#F1F5F9;font-size:2rem;margin:8px 0;">MediPredict AI</h1>
        <p style="color:#64748B;font-size:0.95rem;">Insurance Underwriting & Health Screening Platform</p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1,1.2,1])
    with col:
        tab_login, tab_register = st.tabs(["🔐 Employee Login","📝 Register"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            u = st.text_input("Username", key="lu", placeholder="Employee username")
            p = st.text_input("Password", type="password", key="lp", placeholder="Password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login →", use_container_width=True, type="primary"):
                if not u or not p:
                    st.error("Please fill both fields.")
                else:
                    user = verify_user(u, p)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        st.session_state.ai_risk_scores = {"diabetes":0.0,"heart":0.0,"liver":0.0}
                        st.session_state.active_patient = None
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)
            r_name  = st.text_input("Full Name*",       key="rn",  placeholder="Rahul Sharma")
            r_email = st.text_input("Official Email*",  key="re",  placeholder="rahul@company.com")
            r_eid   = st.text_input("Employee ID*",     key="reid", placeholder="EMP-00123")
            r_dept  = st.selectbox("Department*",       ["Underwriting","Claims","Sales","Medical","Operations","Other"], key="rd")
            r_comp  = st.text_input("Company Name*",    key="rc",  placeholder="LIC / HDFC Life / SBI Life …")
            st.divider()
            r_user  = st.text_input("Choose Username*", key="ru",  placeholder="rahul_sharma")
            r_pass  = st.text_input("Password*",        type="password", key="rp",  placeholder="Min 6 characters")
            r_pass2 = st.text_input("Confirm Password*",type="password", key="rp2", placeholder="Repeat password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", use_container_width=True, type="primary"):
                if not all([r_name,r_email,r_eid,r_comp,r_user,r_pass,r_pass2]):
                    st.error("Please fill all required fields (*)")
                elif len(r_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif r_pass != r_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = create_user(r_user, r_pass, r_name, r_email, r_eid, r_dept, r_comp)
                    st.success(msg + " Please login.") if ok else st.error(msg)

if not st.session_state.logged_in:
    show_auth_page()
    st.stop()


# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    return {
        "diabetes": joblib.load("models/diabetes_model.sav"),
        "heart":    joblib.load("models/heart_disease_model.sav"),
        "liver":    joblib.load("models/liver_model.sav"),
    }

models = load_models()
current_user = st.session_state.user
active_patient = st.session_state.active_patient


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def risk_gauge(probability, title="Risk Score"):
    color = "#EF4444" if probability>=0.66 else "#EAB308" if probability>=0.33 else "#22C55E"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability*100,1),
        number={"suffix":"%","font":{"size":48,"color":color}},
        title={"text":title,"font":{"size":13,"color":"#94A3B8"}},
        gauge={"axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#334155","tickfont":{"color":"#64748B","size":11}},
               "bar":{"color":color,"thickness":0.22},"bgcolor":"#1E293B","borderwidth":0,
               "steps":[{"range":[0,33],"color":"rgba(34,197,94,0.12)"},
                        {"range":[33,66],"color":"rgba(234,179,8,0.12)"},
                        {"range":[66,100],"color":"rgba(239,68,68,0.12)"}],
               "threshold":{"line":{"color":color,"width":3},"thickness":0.8,"value":probability*100}},
    ))
    fig.update_layout(height=240,margin=dict(t=30,b=0,l=20,r=20),
                      paper_bgcolor="#0F172A",plot_bgcolor="#0F172A",font={"color":"#F1F5F9"})
    st.plotly_chart(fig, use_container_width=True)

def risk_badge_html(prob):
    if prob>=0.66: return '<span class="disease-label-pos">🔴 High Risk</span>',"#FCA5A5"
    elif prob>=0.33: return '<span class="disease-label-med">🟡 Moderate Risk</span>',"#FDE047"
    else: return '<span class="disease-label-neg">🟢 Low Risk</span>',"#86EFAC"

def range_indicator(label,value,low,high,unit=""):
    if value==0: return
    if value<low: cls,icon,note="ri-warning","⬇",f"Below normal ({low}–{high} {unit})"
    elif value>high: cls,icon,note="ri-danger","⬆",f"Above normal ({low}–{high} {unit})"
    else: cls,icon,note="ri-normal","✔",f"Within normal range ({low}–{high} {unit})"
    st.markdown(f'<div class="{cls}"><b>{icon} {label}:</b> {value} {unit} · {note}</div>',unsafe_allow_html=True)

def shap_bar_chart(model,input_array,feature_names):
    try:
        explainer=shap.TreeExplainer(model)
        sv_raw=explainer.shap_values(input_array)
        sv=sv_raw[1][0] if isinstance(sv_raw,list) else sv_raw[0]
        df=pd.DataFrame({"Feature":feature_names,"SHAP":sv,"Abs":np.abs(sv)})
        df=df.sort_values("Abs",ascending=True).tail(10)
        clrs=["#EF4444" if v>0 else "#22C55E" for v in df["SHAP"]]
        fig=go.Figure(go.Bar(x=df["SHAP"],y=df["Feature"],orientation="h",marker_color=clrs,
            text=[f"{v:+.3f}" for v in df["SHAP"]],textposition="outside",
            textfont={"size":11,"color":"#CBD5E1"}))
        fig.update_layout(title={"text":"Feature Contributions (SHAP)","font":{"size":13,"color":"#94A3B8"}},
            xaxis={"title":"Impact","titlefont":{"size":11,"color":"#64748B"},"tickfont":{"size":10,"color":"#64748B"},"gridcolor":"#1E293B"},
            yaxis={"tickfont":{"size":11,"color":"#CBD5E1"},"gridcolor":"#1E293B"},
            height=340,margin=dict(l=10,r=60,t=40,b=10),paper_bgcolor="#0F172A",plot_bgcolor="#0F172A")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"SHAP unavailable: {e}")

def show_result_card(result_text,prob,positive):
    badge_html,score_color=risk_badge_html(prob)
    cc="result-positive" if positive else "result-negative"
    tc="result-title-pos" if positive else "result-title-neg"
    emoji="⚠️" if positive else "✅"
    note=("Please consult a qualified healthcare professional." if positive
          else "Continue regular health check-ups and maintain a healthy lifestyle.")
    st.markdown(f"""
    <div class="{cc}"><div class="{tc}">{emoji} {result_text}</div>
    <div style="margin-top:8px;display:flex;align-items:center;gap:12px;">
        {badge_html}
        <span style="font-size:0.88rem;color:#94A3B8;">Risk Score: <b style="color:{score_color}">{prob*100:.1f}%</b></span>
    </div><div class="result-sub">{note}</div></div>""", unsafe_allow_html=True)

def show_recommendations(recs,positive):
    st.subheader("💡 Recommendations")
    for i,rec in enumerate(recs):
        cls="rec-card rec-card-urgent" if (positive and i==0) else "rec-card"
        prefix="🚨 " if (positive and i==0) else "• "
        st.markdown(f'<div class="{cls}">{prefix}{rec}</div>',unsafe_allow_html=True)

def patient_banner():
    """Show active patient strip at top of screening pages"""
    ap = st.session_state.active_patient
    if ap:
        st.markdown(f"""
        <div style="background:rgba(29,78,216,0.15);border:1px solid rgba(59,130,246,0.4);
                    border-radius:10px;padding:12px 20px;margin-bottom:16px;
                    display:flex;align-items:center;gap:16px;">
            <span style="font-size:1.4rem;">👤</span>
            <div>
                <span style="color:#93C5FD;font-weight:700;font-size:1rem;">{ap['patient_name']}</span>
                <span style="color:#475569;font-size:0.82rem;margin-left:12px;">ID: {ap['patient_id']}</span>
                <span style="color:#475569;font-size:0.82rem;margin-left:12px;">📞 {ap.get('phone','—')}</span>
                <span style="color:#475569;font-size:0.82rem;margin-left:12px;">Policy: {ap.get('policy_number','—')}</span>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No patient selected. Go to **Patient Management** to select or register a patient first.")

def generate_pdf(disease, inputs, probability, result, recommendations, patient=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    story = []
    from datetime import datetime as _dt
    header_style = ParagraphStyle("hdr", fontSize=18, fontName="Helvetica-Bold",
                                   textColor=colors.white, backColor=colors.HexColor("#0F172A"),
                                   alignment=TA_CENTER, spaceAfter=4, leading=28,
                                   leftIndent=-1*cm, rightIndent=-1*cm, borderPadding=12)
    story.append(Paragraph("MediPredict AI — Health Screening Report", header_style))
    sub_style = ParagraphStyle("sub",fontSize=9,fontName="Helvetica",
                                textColor=colors.HexColor("#94A3B8"),alignment=TA_CENTER,spaceAfter=12)
    story.append(Paragraph(f"Generated: {_dt.now().strftime('%B %d, %Y at %H:%M')}  |  "
                           f"Employee: {current_user.get('full_name','N/A')} ({current_user.get('employee_id','')})",sub_style))

    # Patient info box
    if patient:
        label_style = ParagraphStyle("lbl",fontSize=10,fontName="Helvetica-Bold",
                                      textColor=colors.HexColor("#1D4ED8"),spaceAfter=4,spaceBefore=8)
        story.append(Paragraph("Patient Information", label_style))
        pt_data = [
            ["Patient Name", patient.get("patient_name","—"), "Patient ID", patient.get("patient_id","—")],
            ["Phone", patient.get("phone","—"), "DOB", patient.get("date_of_birth","—")],
            ["Policy No.", patient.get("policy_number","—"), "Gender", patient.get("gender","—")],
            ["City", patient.get("city","—"), "Nominee", patient.get("nominee_name","—")],
        ]
        pt = Table(pt_data, colWidths=[4*cm,5*cm,4*cm,5*cm])
        pt.setStyle(TableStyle([
            ("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),
            ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
            ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#334155")),
            ("TEXTCOLOR",(2,0),(2,-1),colors.HexColor("#334155")),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#EFF6FF"),colors.white]),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#BFDBFE")),
            ("LEFTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ]))
        story.append(pt)
        story.append(Spacer(1,10))

    risk_color = (colors.HexColor("#EF4444") if probability>=0.66 else
                  colors.HexColor("#EAB308") if probability>=0.33 else colors.HexColor("#22C55E"))
    story.append(Paragraph(f"Disease Screened: {disease}",
                            ParagraphStyle("lbl2",fontSize=11,fontName="Helvetica-Bold",
                                           textColor=colors.HexColor("#0F172A"),spaceAfter=4)))
    story.append(Paragraph(f"Result: {result}   |   Risk Score: {probability*100:.1f}%",
                            ParagraphStyle("res",fontSize=12,fontName="Helvetica-Bold",
                                           textColor=risk_color,spaceAfter=12)))
    story.append(Paragraph("Input Parameters:",
                            ParagraphStyle("h2",fontSize=11,fontName="Helvetica-Bold",
                                           textColor=colors.HexColor("#0F172A"),spaceAfter=6,spaceBefore=8)))
    t = Table([[str(k),str(v)] for k,v in inputs.items()], colWidths=[6*cm,10*cm])
    t.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),
        ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#334155")),
        ("TEXTCOLOR",(1,0),(1,-1),colors.HexColor("#475569")),
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F8FAFC"),colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#E2E8F0")),
        ("LEFTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(t)
    if recommendations:
        story.append(Spacer(1,12))
        story.append(Paragraph("Recommendations:",ParagraphStyle("h2",fontSize=11,fontName="Helvetica-Bold",
                                textColor=colors.HexColor("#0F172A"),spaceAfter=6)))
        for i,rec in enumerate(recommendations):
            story.append(Paragraph(f"{'>> ' if i==0 else '-  '}{rec}",
                ParagraphStyle("rec",fontSize=9,fontName="Helvetica",
                               textColor=colors.HexColor("#334155"),leftIndent=10,spaceAfter=3)))
    story.append(Spacer(1,16))
    story.append(Paragraph(
        "DISCLAIMER: AI-generated for informational purposes only. Not a medical diagnosis. Consult a healthcare professional.",
        ParagraphStyle("disc",fontSize=7.5,fontName="Helvetica-Oblique",
                       textColor=colors.HexColor("#94A3B8"),borderColor=colors.HexColor("#E2E8F0"),
                       borderWidth=1,borderPadding=8,spaceAfter=0)))
    doc.build(story)
    buf.seek(0)
    return buf

def generate_insurance_pdf(patient_info, risk_result, flags, patient=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    story = []
    from datetime import datetime as _dt
    header_style = ParagraphStyle("hdr",fontSize=18,fontName="Helvetica-Bold",
                                   textColor=colors.white,backColor=colors.HexColor("#0F172A"),
                                   alignment=TA_CENTER,spaceAfter=4,leading=28,
                                   leftIndent=-1*cm,rightIndent=-1*cm,borderPadding=12)
    story.append(Paragraph("MediPredict AI — Insurance Underwriting Report", header_style))
    sub_style = ParagraphStyle("sub",fontSize=9,fontName="Helvetica",
                                textColor=colors.HexColor("#94A3B8"),alignment=TA_CENTER,spaceAfter=12)
    story.append(Paragraph(
        f"Generated: {_dt.now().strftime('%B %d, %Y at %H:%M')}  |  "
        f"Employee: {current_user.get('full_name','N/A')} ({current_user.get('employee_id','')})  |  "
        f"{current_user.get('company','')} — {current_user.get('department','')}",sub_style))

    # Patient details
    if patient:
        label_style = ParagraphStyle("lbl",fontSize=10,fontName="Helvetica-Bold",
                                      textColor=colors.HexColor("#1D4ED8"),spaceAfter=4,spaceBefore=8)
        story.append(Paragraph("Patient / Proposer Details", label_style))
        pt_data = [
            ["Patient Name", patient.get("patient_name","—"), "Patient ID", patient.get("patient_id","—")],
            ["Phone", patient.get("phone","—"), "Email", patient.get("email","—")],
            ["Date of Birth", patient.get("date_of_birth","—"), "Gender", patient.get("gender","—")],
            ["Policy Number", patient.get("policy_number","—"), "Nominee", patient.get("nominee_name","—")],
            ["City", patient.get("city","—"), "Pincode", patient.get("pincode","—")],
            ["Address", patient.get("address","—"), "", ""],
        ]
        pt = Table(pt_data, colWidths=[4*cm,5*cm,4*cm,5*cm])
        pt.setStyle(TableStyle([
            ("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),
            ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
            ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#334155")),
            ("TEXTCOLOR",(2,0),(2,-1),colors.HexColor("#334155")),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#EFF6FF"),colors.white]),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#BFDBFE")),
            ("LEFTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ]))
        story.append(pt)
        story.append(Spacer(1,12))

    risk_color = colors.HexColor(
        "#7F1D1D" if risk_result["riskPoints"]>150 else
        "#EF4444" if risk_result["riskPoints"]>110 else
        "#F97316" if risk_result["riskPoints"]>75 else
        "#EAB308" if risk_result["riskPoints"]>45 else "#22C55E")
    story.append(Paragraph(f"Underwriting Decision: {risk_result['decision']}",
                            ParagraphStyle("res",fontSize=13,fontName="Helvetica-Bold",
                                           textColor=risk_color,spaceAfter=6)))
    story.append(Paragraph(
        f"Risk Category: {risk_result['category']}   |   Total Risk Points: {risk_result['riskPoints']}",
        ParagraphStyle("lbl2",fontSize=11,fontName="Helvetica-Bold",
                       textColor=colors.HexColor("#0F172A"),spaceAfter=4)))
    if risk_result["finalPremium"]:
        story.append(Paragraph(
            f"Estimated Monthly Premium: Rs.{risk_result['finalPremium']:,}   |   Loading: {risk_result['premiumMultiplier']:.2f}x",
            ParagraphStyle("lbl2",fontSize=11,fontName="Helvetica-Bold",
                           textColor=colors.HexColor("#0F172A"),spaceAfter=12)))

    story.append(Paragraph("Health Profile Summary:",
                            ParagraphStyle("h2",fontSize=11,fontName="Helvetica-Bold",
                                           textColor=colors.HexColor("#0F172A"),spaceAfter=6,spaceBefore=8)))
    t = Table([[str(k),str(v)] for k,v in patient_info.items()], colWidths=[6*cm,10*cm])
    t.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),
        ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#334155")),
        ("TEXTCOLOR",(1,0),(1,-1),colors.HexColor("#475569")),
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F8FAFC"),colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#E2E8F0")),
        ("LEFTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(t)

    if flags:
        story.append(Spacer(1,12))
        story.append(Paragraph("Underwriting Risk Flags:",
                                ParagraphStyle("h2",fontSize=11,fontName="Helvetica-Bold",
                                               textColor=colors.HexColor("#0F172A"),spaceAfter=6)))
        sev_clr={"critical":"#EF4444","high":"#F97316","medium":"#EAB308","low":"#22C55E"}
        for f in flags:
            story.append(Paragraph(f"[{f['severity'].upper()}] {f['label']}",
                ParagraphStyle("flag",fontSize=9,fontName="Helvetica",
                               textColor=colors.HexColor(sev_clr.get(f["severity"],"#94A3B8")),
                               leftIndent=10,spaceAfter=3)))
    story.append(Spacer(1,16))
    story.append(Paragraph(
        "DISCLAIMER: AI-generated for actuarial support only. Final decisions must be made by a licensed underwriter.",
        ParagraphStyle("disc",fontSize=7.5,fontName="Helvetica-Oblique",
                       textColor=colors.HexColor("#94A3B8"),borderColor=colors.HexColor("#E2E8F0"),
                       borderWidth=1,borderPadding=8)))
    doc.build(story)
    buf.seek(0)
    return buf


RECOMMENDATIONS = {
    "diabetes":{
        True:["Consult an endocrinologist immediately.",
              "Monitor blood glucose daily; target fasting glucose < 100 mg/dL.",
              "Adopt a low-glycemic diet: whole grains, legumes, leafy greens.",
              "Aim for 150 min/week of moderate aerobic exercise.",
              "Reduce sugary beverages and refined carbohydrates.",
              "Maintain BMI between 18.5 and 24.9."],
        False:["Maintain a balanced diet rich in fibre and low in sugar.",
               "Exercise regularly to keep insulin sensitivity high.",
               "Monitor glucose annually if you have a family history.",
               "Stay hydrated and limit alcohol intake."]},
    "heart":{
        True:["Seek immediate cardiac evaluation from a cardiologist.",
              "Monitor blood pressure; target < 120/80 mmHg.",
              "Follow a heart-healthy diet: reduce saturated fats, salt, red meat.",
              "Quit smoking — it is the #1 modifiable risk factor.",
              "Take prescribed medications consistently.",
              "Limit alcohol to 1-2 drinks per day maximum."],
        False:["Keep cholesterol in check (LDL < 100 mg/dL).",
               "Exercise for at least 30 minutes most days.",
               "Manage stress through mindfulness or yoga.",
               "Get regular blood pressure and cholesterol screenings."]},
    "liver":{
        True:["Consult a hepatologist or gastroenterologist.",
              "Avoid alcohol completely.",
              "Follow a low-fat, high-fibre diet.",
              "Stay well-hydrated.",
              "Avoid excess OTC medications that burden the liver.",
              "Get vaccinated against Hepatitis A and B."],
        False:["Limit alcohol consumption.",
               "Maintain a healthy weight to prevent fatty liver.",
               "Eat plenty of antioxidant-rich foods.",
               "Get annual liver function tests if you have risk factors."]},
}


# ══════════════════════════════════════════════════════
# INSURANCE ENGINE
# ══════════════════════════════════════════════════════

def compute_insurance_risk(age,bmi,smoker,alcohol,occupation,family_history,
                            existing_conditions,d_risk,h_risk,l_risk):
    rp=0; flags=[]
    age_pts={(0,25):0,(25,35):5,(35,45):10,(45,55):20,(55,65):35}
    for (lo,hi),pts in age_pts.items():
        if lo<=age<hi: rp+=pts; break
    else:
        rp+=55; flags.append({"label":"Senior Age Group (65+)","severity":"medium"})
    if bmi<18.5:   rp+=10;  flags.append({"label":f"Underweight (BMI {bmi:.1f})","severity":"medium"})
    elif bmi<=24.9: pass
    elif bmi<=29.9: rp+=10; flags.append({"label":f"Overweight (BMI {bmi:.1f})","severity":"low"})
    elif bmi<=34.9: rp+=25; flags.append({"label":f"Obese Class I (BMI {bmi:.1f})","severity":"medium"})
    else:           rp+=45; flags.append({"label":f"Obese Class II+ (BMI {bmi:.1f})","severity":"high"})
    if smoker=="Current Smoker":  rp+=40; flags.append({"label":"Current Smoker","severity":"high"})
    elif smoker=="Former Smoker": rp+=15; flags.append({"label":"Former Smoker","severity":"medium"})
    if alcohol=="Heavy":    rp+=30; flags.append({"label":"Heavy Alcohol Use","severity":"high"})
    elif alcohol=="Moderate": rp+=8
    if occupation=="Hazardous":      rp+=30; flags.append({"label":"Hazardous Occupation","severity":"medium"})
    elif occupation=="Moderate Risk": rp+=10
    for cond,pts in {"Heart Disease":15,"Cancer":15,"Diabetes":10,"Stroke":12}.items():
        if cond in family_history:
            rp+=pts; flags.append({"label":f"Family History: {cond}","severity":"medium"})
    for cond,(pts,sev) in {"Hypertension":(25,"medium"),"Diabetes":(35,"high"),
                            "Heart Disease":(50,"high"),"Cancer":(80,"critical"),
                            "Liver Disease":(30,"high"),"Kidney Disease":(35,"high")}.items():
        if cond in existing_conditions:
            rp+=pts; flags.append({"label":f"Existing: {cond}","severity":sev})
    def ai_flag(name,risk_pct,ph,pm,sh,sm):
        nonlocal rp
        r=risk_pct/100
        if r>0.66:   rp+=ph; flags.append({"label":f"AI: High {name} Risk ({risk_pct:.0f}%)","severity":sh})
        elif r>0.33: rp+=pm; flags.append({"label":f"AI: Moderate {name} Risk ({risk_pct:.0f}%)","severity":sm})
    ai_flag("Diabetes",d_risk,35,15,"high","medium")
    ai_flag("Heart Disease",h_risk,40,18,"high","medium")
    ai_flag("Liver Disease",l_risk,30,12,"high","medium")
    if rp<=20:   cat,dec,mult="Preferred","✅ Accept — Preferred Rates",0.85; note="Excellent profile. Full coverage with premium discount."; col="#22C55E"
    elif rp<=45: cat,dec,mult="Standard","✅ Accept — Standard Rates",1.00; note="Good profile. Full coverage at standard premium."; col="#86EFAC"
    elif rp<=75: cat,dec,mult="Substandard A","⚠️ Accept — Extra Loading (35%)",1.35; note="Moderate risk. Extra loading applied."; col="#FDE047"
    elif rp<=110:cat,dec,mult="Substandard B","⚠️ Accept — High Loading (75%)",1.75; note="High risk. Significant loading required."; col="#F97316"
    elif rp<=150:cat,dec,mult="Rated / Referred","🔴 Conditional — Underwriter Review",2.25; note="Very high risk. Limited coverage only."; col="#EF4444"
    else:        cat,dec,mult="Declined","⛔ Decline — Uninsurable",None; note="Risk exceeds limits. Re-apply after 2 years."; col="#7F1D1D"
    base=800 if age<35 else 1500 if age<50 else 2800
    final_premium=round(base*mult) if mult else None
    return {"riskPoints":rp,"category":cat,"decision":dec,"premiumMultiplier":mult,
            "finalPremium":final_premium,"coverageNote":note,"decisionColor":col,"flags":flags}

def show_insurance_flag(flag):
    cls=f"ins-flag-{flag['severity']}"
    icons={"critical":"⛔","high":"🔴","medium":"🟡","low":"🟢"}
    bc={"critical":"#EF4444","high":"#F97316","medium":"#EAB308","low":"#22C55E"}.get(flag["severity"],"#94A3B8")
    st.markdown(f'<div class="{cls}">{icons.get(flag["severity"],"•")} {flag["label"]}'
                f'<span style="float:right;font-size:0.72rem;color:{bc};font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.5px">{flag["severity"]}</span></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏥 MediPredict AI")
    st.caption(f"👤 **{current_user.get('full_name', current_user['username'])}**")
    st.caption(f"🏢 {current_user.get('company','—')} · {current_user.get('department','—')}")
    st.caption(f"🪪 {current_user.get('employee_id','—')}")
    st.divider()

    selected = option_menu(
        menu_title=None,
        options=["Patient Management","General Disease","Diabetes","Heart Disease",
                 "Liver Disease","Insurance Report","Patient History","My Account"],
        icons=["people","activity","droplet","heart","person","shield-check","clock-history","person-circle"],
        default_index=0,
        styles={
            "container":         {"padding":"0","background-color":"transparent"},
            "icon":              {"color":"#94A3B8","font-size":"15px"},
            "nav-link":          {"font-size":"13px","color":"#CBD5E1","padding":"9px 14px",
                                  "border-radius":"8px","--hover-color":"rgba(59,130,246,0.15)"},
            "nav-link-selected": {"background-color":"rgba(59,130,246,0.25)",
                                  "color":"#93C5FD","font-weight":"600"},
        }
    )

    st.divider()
    ap = st.session_state.active_patient
    if ap:
        st.caption("👤 Active Patient")
        st.caption(f"**{ap['patient_name']}**")
        st.caption(f"ID: {ap['patient_id']}")
        scores = st.session_state.ai_risk_scores
        if any(v>0 for v in scores.values()):
            st.caption("🤖 AI Risk Scores")
            for k,v in scores.items():
                if v>0:
                    ic = "🔴" if v>=66 else "🟡" if v>=33 else "🟢"
                    st.caption(f"{ic} {k.title()}: **{v:.1f}%**")
        st.divider()

    if st.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in","user","ai_risk_scores","active_patient"]:
            st.session_state[k] = False if k=="logged_in" else None if k in ["user","active_patient"] else {"diabetes":0.0,"heart":0.0,"liver":0.0}
        st.rerun()

    st.warning("**⚠️ Disclaimer**\nFor informational purposes only.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Patient Management":
    st.title("👥 Patient Management")
    st.caption("Register new patients, search existing records, and select a patient to begin screening.")
    st.divider()

    tab_list, tab_register, tab_edit = st.tabs(["📋 Patient Records","➕ Register New Patient","✏️ Edit Patient"])

    with tab_list:
        patients = get_patients(current_user["id"])
        if not patients:
            st.info("No patients registered yet. Use **Register New Patient** tab to add patients.")
        else:
            # Search bar
            search = st.text_input("🔍 Search by name, ID, phone or policy number",
                                    placeholder="Type to filter…", key="pt_search")
            filtered = [p for p in patients if not search or any(
                search.lower() in str(p.get(f,"")).lower()
                for f in ["patient_name","patient_id","phone","policy_number"]
            )]

            st.caption(f"Showing {len(filtered)} of {len(patients)} patients")
            for p in filtered:
                is_active = st.session_state.active_patient and st.session_state.active_patient["id"]==p["id"]
                card_class = "patient-card-selected" if is_active else "patient-card"
                active_badge = ' <span style="background:#1D4ED8;color:white;padding:2px 10px;border-radius:20px;font-size:0.72rem;margin-left:8px;">ACTIVE</span>' if is_active else ""
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                        <div>
                            <span style="color:#F1F5F9;font-weight:700;font-size:1rem;">{p['patient_name']}</span>{active_badge}
                            <span style="color:#475569;font-size:0.8rem;margin-left:12px;">ID: {p['patient_id']}</span>
                        </div>
                        <div style="color:#64748B;font-size:0.8rem;">
                            📞 {p.get('phone','—')} &nbsp;|&nbsp; 
                            🎂 {p.get('date_of_birth','—')} &nbsp;|&nbsp;
                            📋 Policy: {p.get('policy_number','—')} &nbsp;|&nbsp;
                            📍 {p.get('city','—')}
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                c1, c2, c3 = st.columns([2,2,1])
                with c1:
                    if st.button(f"✅ Select as Active Patient", key=f"sel_{p['id']}", use_container_width=True):
                        st.session_state.active_patient = p
                        st.session_state.ai_risk_scores = {"diabetes":0.0,"heart":0.0,"liver":0.0}
                        st.success(f"✅ {p['patient_name']} set as active patient!")
                        st.rerun()
                with c2:
                    scr_count = len(get_screenings_db(current_user["id"], p["id"]))
                    ins_count = len(get_insurance_reports_db(current_user["id"], p["id"]))
                    st.caption(f"🩺 {scr_count} screenings &nbsp; 🛡️ {ins_count} reports")
                with c3:
                    if st.button("🗑️ Delete", key=f"del_{p['id']}", use_container_width=True):
                        delete_patient(p["id"], current_user["id"])
                        if st.session_state.active_patient and st.session_state.active_patient["id"]==p["id"]:
                            st.session_state.active_patient = None
                        st.rerun()

    with tab_register:
        st.subheader("📝 Register New Patient")
        c1, c2 = st.columns(2)
        with c1:
            np_name    = st.text_input("Patient Full Name*",   key="np_name",  placeholder="Ramesh Kumar")
            np_id      = st.text_input("Patient ID*",          key="np_id",    placeholder="PAT-2025-001")
            np_phone   = st.text_input("Phone Number*",        key="np_phone", placeholder="+91 98765 43210")
            np_email   = st.text_input("Email Address",        key="np_email", placeholder="ramesh@email.com")
            np_dob     = st.text_input("Date of Birth*",       key="np_dob",   placeholder="DD/MM/YYYY")
            np_gender  = st.selectbox("Gender*",               ["Male","Female","Other"], key="np_gender")
        with c2:
            np_policy  = st.text_input("Policy Number",        key="np_policy", placeholder="POL-2025-XXXXX")
            np_nominee = st.text_input("Nominee Name",         key="np_nominee",placeholder="Sunita Kumar")
            np_addr    = st.text_area("Address",               key="np_addr",   placeholder="House No., Street, Area",height=80)
            np_city    = st.text_input("City",                 key="np_city",   placeholder="Mumbai")
            np_pin     = st.text_input("Pincode",              key="np_pin",    placeholder="400001")
            np_notes   = st.text_area("Notes / Remarks",       key="np_notes",  placeholder="Any special notes…",height=80)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Register Patient", use_container_width=True, type="primary"):
            if not all([np_name, np_id, np_phone, np_dob]):
                st.error("Please fill all required fields (*): Name, Patient ID, Phone, DOB.")
            else:
                ok, msg = create_patient(current_user["id"], np_name, np_id, np_phone, np_email,
                                          np_dob, np_gender, np_addr, np_city, np_pin,
                                          np_policy, np_nominee, np_notes)
                if ok:
                    st.success(f"✅ {msg} You can now select {np_name} from the Patient Records tab.")
                else:
                    st.error(msg)

    with tab_edit:
        patients2 = get_patients(current_user["id"])
        if not patients2:
            st.info("No patients to edit yet.")
        else:
            edit_options = {f"{p['patient_name']} ({p['patient_id']})": p["id"] for p in patients2}
            edit_choice = st.selectbox("Select Patient to Edit", list(edit_options.keys()), key="edit_sel")
            if edit_choice:
                ep = get_patient_by_id(edit_options[edit_choice])
                if ep:
                    c1, c2 = st.columns(2)
                    with c1:
                        e_name    = st.text_input("Patient Full Name*", value=ep.get("patient_name",""),   key="e_name")
                        e_phone   = st.text_input("Phone Number*",      value=ep.get("phone",""),          key="e_phone")
                        e_email   = st.text_input("Email Address",      value=ep.get("email",""),          key="e_email")
                        e_dob     = st.text_input("Date of Birth",      value=ep.get("date_of_birth",""),  key="e_dob")
                        e_gender  = st.selectbox("Gender", ["Male","Female","Other"],
                                                  index=["Male","Female","Other"].index(ep.get("gender","Male")) if ep.get("gender") in ["Male","Female","Other"] else 0,
                                                  key="e_gender")
                    with c2:
                        e_policy  = st.text_input("Policy Number",  value=ep.get("policy_number",""),   key="e_policy")
                        e_nominee = st.text_input("Nominee Name",   value=ep.get("nominee_name",""),    key="e_nominee")
                        e_addr    = st.text_area("Address",         value=ep.get("address",""),         key="e_addr", height=80)
                        e_city    = st.text_input("City",           value=ep.get("city",""),            key="e_city")
                        e_pin     = st.text_input("Pincode",        value=ep.get("pincode",""),         key="e_pin")
                        e_notes   = st.text_area("Notes",           value=ep.get("notes",""),           key="e_notes", height=80)
                    st.markdown("<br>",unsafe_allow_html=True)
                    if st.button("💾 Save Changes", use_container_width=True, type="primary"):
                        update_patient(ep["id"],e_name,e_phone,e_email,e_dob,e_gender,
                                       e_addr,e_city,e_pin,e_policy,e_nominee,e_notes)
                        if st.session_state.active_patient and st.session_state.active_patient["id"]==ep["id"]:
                            st.session_state.active_patient = get_patient_by_id(ep["id"])
                        st.success("✅ Patient record updated successfully!")
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GENERAL DISEASE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "General Disease":
    st.title("🔬 General Disease Prediction")
    patient_banner()
    st.divider()
    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')
    symptoms = st.multiselect("Select all symptoms you are experiencing:",
                               options=disease_model.all_symptoms, placeholder="Type to search symptoms...")
    X = prepare_symptoms_array(symptoms)
    if st.button("🔍 Run Prediction", use_container_width=True):
        if not symptoms:
            st.warning("Please select at least one symptom to continue.")
        else:
            prediction, prob = disease_model.predict(X)
            badge_html, score_color = risk_badge_html(prob)
            col1, col2 = st.columns(2)
            with col1: risk_gauge(prob, "Confidence Score")
            with col2:
                st.markdown(f"""
                <div class="disease-card">
                    <div style="color:#64748B;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Predicted Condition</div>
                    <div class="disease-name">{prediction}</div>
                    {badge_html}
                    <div style="margin-top:14px;color:#64748B;font-size:0.85rem;">Confidence: <b style="color:{score_color}">{prob*100:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)
            st.divider()
            tab1, tab2 = st.tabs(["📄 Description","🛡️ Precautions"])
            with tab1: st.info(disease_model.describe_predicted_disease())
            with tab2:
                precautions = disease_model.predicted_disease_precautions()
                for i in range(4):
                    st.markdown(f'<div class="rec-card"><b>{i+1}.</b> {precautions[i]}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DIABETES
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Diabetes":
    st.title("💉 Diabetes Risk Assessment")
    patient_banner()
    st.divider()
    with st.expander("📊 Normal Reference Ranges"):
        c1,c2=st.columns(2)
        c1.markdown("**Glucose (fasting):** 70–99 mg/dL  \n**Blood Pressure:** 60–80 mmHg")
        c2.markdown("**BMI:** 18.5–24.9 kg/m²  \n**Insulin:** 2–25 μIU/mL")
    st.subheader("📝 Patient Parameters")
    col1,col2,col3=st.columns(3)
    with col1:
        Pregnancies=st.number_input("Pregnancies",min_value=0)
        SkinThickness=st.number_input("Skin Thickness (mm)",min_value=0)
        DiabetesPedigreeFunction=st.number_input("Diabetes Pedigree Function",min_value=0.0,step=0.01)
    with col2:
        Glucose=st.number_input("Glucose (mg/dL)",min_value=0)
        Insulin=st.number_input("Insulin (μIU/mL)",min_value=0)
        Age=st.number_input("Age (years)",min_value=0)
    with col3:
        BloodPressure=st.number_input("Blood Pressure (mmHg)",min_value=0)
        BMI=st.number_input("BMI (kg/m²)",min_value=0.0,step=0.1)
    st.subheader("🩺 Input Validation")
    c1,c2=st.columns(2)
    with c1:
        range_indicator("Glucose",Glucose,70,99,"mg/dL")
        range_indicator("Blood Pressure",BloodPressure,60,80,"mmHg")
    with c2:
        range_indicator("BMI",BMI,18.5,24.9,"kg/m²")
        range_indicator("Insulin",Insulin,2,25,"μIU/mL")
    input_array=np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    feature_names=["Pregnancies","Glucose","Blood Pressure","Skin Thickness","Insulin","BMI","Diabetes Pedigree","Age"]
    st.divider()
    if st.button("🔍 Run Diabetes Screening", use_container_width=True):
        prob=models["diabetes"].predict_proba(input_array)[0][1]
        prediction=int(prob>=0.5)
        result="Diabetic" if prediction else "Not Diabetic"
        recs=RECOMMENDATIONS["diabetes"][bool(prediction)]
        inputs=dict(zip(feature_names,input_array[0].tolist()))
        st.session_state.ai_risk_scores["diabetes"]=round(prob*100,1)
        col1,col2=st.columns(2)
        with col1: risk_gauge(prob,"Diabetes Risk Score")
        with col2: show_result_card(result,prob,bool(prediction))
        col3,col4=st.columns(2)
        with col3: shap_bar_chart(models["diabetes"],input_array,feature_names)
        with col4: show_recommendations(recs,bool(prediction))
        ap=st.session_state.active_patient
        pid=ap["id"] if ap else None
        save_screening_db(current_user["id"],pid,"Diabetes",result,round(prob*100,1),inputs)
        st.info("💡 Risk score saved! Go to **Insurance Report** for underwriting analysis.")
        pdf_buf=generate_pdf("Diabetes",inputs,prob,result,recs,ap)
        st.download_button("📥 Download PDF Report",pdf_buf,"diabetes_report.pdf","application/pdf",use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HEART DISEASE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Heart Disease":
    st.title("❤️ Heart Disease Risk Assessment")
    patient_banner()
    st.divider()
    with st.expander("📊 Normal Reference Ranges"):
        c1,c2=st.columns(2)
        c1.markdown("**Blood Pressure:** 90–120 mmHg  \n**Cholesterol:** < 200 mg/dL")
        c2.markdown("**Max Heart Rate:** 100–170 bpm  \n**ST Depression:** 0–1")
    st.subheader("📝 Patient Parameters")
    col1,col2,col3=st.columns(3)
    with col1:
        age=st.number_input("Age (years)",min_value=0)
        trestbps=st.number_input("Resting Blood Pressure (mmHg)",min_value=0)
        thalach=st.number_input("Max Heart Rate Achieved",min_value=0)
        ca=st.number_input("Major Vessels (0–3)",min_value=0,max_value=3)
    with col2:
        sex_opts=("Male","Female")
        sex_val=st.selectbox("Gender",[0,1],format_func=lambda x:sex_opts[x])
        sex=1 if sex_val==0 else 0
        chol=st.number_input("Serum Cholesterol (mg/dL)",min_value=0)
        oldpeak=st.number_input("ST Depression (Oldpeak)",min_value=0.0,step=0.1)
        thal_opts=("Normal","Fixed Defect","Reversible Defect")
        thal=st.selectbox("Thalassemia",[0,1,2],format_func=lambda x:thal_opts[x])
    with col3:
        cp_opts=("Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptotic")
        cp=st.selectbox("Chest Pain Type",[0,1,2,3],format_func=lambda x:cp_opts[x])
        ecg_opts=("Normal","ST-T Abnormality","Left Ventricular Hypertrophy")
        restecg=st.selectbox("Resting ECG",[0,1,2],format_func=lambda x:ecg_opts[x])
        slope_opts=("Upsloping","Flat","Downsloping")
        slope=st.selectbox("Peak Exercise ST Slope",[0,1,2],format_func=lambda x:slope_opts[x])
        exang=1 if st.checkbox("Exercise Induced Angina") else 0
        fbs=1 if st.checkbox("Fasting Blood Sugar > 120 mg/dL") else 0
    st.subheader("🩺 Input Validation")
    c1,c2,c3=st.columns(3)
    with c1: range_indicator("Blood Pressure",trestbps,90,120,"mmHg")
    with c2: range_indicator("Cholesterol",chol,0,200,"mg/dL")
    with c3: range_indicator("Max Heart Rate",thalach,60,100,"bpm")
    input_array=np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    feature_names=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
    inputs={"Age":age,"Sex":sex_opts[sex_val],"Chest Pain":cp_opts[cp],"BP":trestbps,"Cholesterol":chol,
            "FBS":fbs,"ECG":ecg_opts[restecg],"Max HR":thalach,"Ex Angina":exang,"Oldpeak":oldpeak,
            "Slope":slope_opts[slope],"Vessels":ca,"Thal":thal_opts[thal]}
    st.divider()
    if st.button("🔍 Run Heart Disease Screening", use_container_width=True):
        prob=models["heart"].predict_proba(input_array)[0][1]
        prediction=int(prob>=0.5)
        result="Heart Disease Detected" if prediction else "No Heart Disease Detected"
        recs=RECOMMENDATIONS["heart"][bool(prediction)]
        st.session_state.ai_risk_scores["heart"]=round(prob*100,1)
        col1,col2=st.columns(2)
        with col1: risk_gauge(prob,"Heart Disease Risk Score")
        with col2: show_result_card(result,prob,bool(prediction))
        col3,col4=st.columns(2)
        with col3: shap_bar_chart(models["heart"],input_array,feature_names)
        with col4: show_recommendations(recs,bool(prediction))
        ap=st.session_state.active_patient
        pid=ap["id"] if ap else None
        save_screening_db(current_user["id"],pid,"Heart Disease",result,round(prob*100,1),inputs)
        st.info("💡 Risk score saved! Go to **Insurance Report** for underwriting analysis.")
        pdf_buf=generate_pdf("Heart Disease",inputs,prob,result,recs,ap)
        st.download_button("📥 Download PDF Report",pdf_buf,"heart_report.pdf","application/pdf",use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVER DISEASE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Liver Disease":
    st.title("🫀 Liver Disease Risk Assessment")
    patient_banner()
    st.divider()
    with st.expander("📊 Normal Reference Ranges"):
        c1,c2=st.columns(2)
        c1.markdown("**Total Bilirubin:** 0.1–1.2 mg/dL  \n**Direct Bilirubin:** 0.0–0.3 mg/dL  \n**Alk Phosphotase:** 44–147 IU/L  \n**ALT:** 7–56 IU/L")
        c2.markdown("**AST:** 10–40 IU/L  \n**Total Proteins:** 6.3–8.2 g/dL  \n**Albumin:** 3.5–5.0 g/dL  \n**A/G Ratio:** 1.0–2.5")
    st.subheader("📝 Patient Parameters")
    col1,col2,col3=st.columns(3)
    with col1:
        sex_opts=("Male","Female")
        sex_val=st.selectbox("Gender",[0,1],format_func=lambda x:sex_opts[x])
        Sex=0 if sex_val==0 else 1
        age=st.number_input("Age (years)",min_value=0)
        Total_Bilirubin=st.number_input("Total Bilirubin (mg/dL)",min_value=0.0,step=0.1)
        Direct_Bilirubin=st.number_input("Direct Bilirubin (mg/dL)",min_value=0.0,step=0.1)
    with col2:
        Alkaline_Phosphotase=st.number_input("Alkaline Phosphotase (IU/L)",min_value=0)
        Alamine_Aminotransferase=st.number_input("ALT (IU/L)",min_value=0)
        Aspartate_Aminotransferase=st.number_input("AST (IU/L)",min_value=0)
    with col3:
        Total_Protiens=st.number_input("Total Proteins (g/dL)",min_value=0.0,step=0.1)
        Albumin=st.number_input("Albumin (g/dL)",min_value=0.0,step=0.1)
        Albumin_and_Globulin_Ratio=st.number_input("Albumin/Globulin Ratio",min_value=0.0,step=0.01)
    st.subheader("🩺 Input Validation")
    c1,c2=st.columns(2)
    with c1:
        range_indicator("Total Bilirubin",Total_Bilirubin,0.1,1.2,"mg/dL")
        range_indicator("Direct Bilirubin",Direct_Bilirubin,0.0,0.3,"mg/dL")
        range_indicator("Alk Phosphotase",Alkaline_Phosphotase,44,147,"IU/L")
    with c2:
        range_indicator("ALT",Alamine_Aminotransferase,7,56,"IU/L")
        range_indicator("AST",Aspartate_Aminotransferase,10,40,"IU/L")
        range_indicator("Albumin",Albumin,3.5,5.0,"g/dL")
    input_array=np.array([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,
                            Alamine_Aminotransferase,Aspartate_Aminotransferase,
                            Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])
    feature_names=["Sex","Age","Total Bilirubin","Direct Bilirubin","Alk Phosphotase",
                   "ALT","AST","Total Proteins","Albumin","A/G Ratio"]
    inputs=dict(zip(feature_names,[sex_opts[sex_val],age,Total_Bilirubin,Direct_Bilirubin,
                                    Alkaline_Phosphotase,Alamine_Aminotransferase,
                                    Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]))
    st.divider()
    if st.button("🔍 Run Liver Disease Screening", use_container_width=True):
        prob=models["liver"].predict_proba(input_array)[0][1]
        prediction=int(prob>=0.5)
        result="Liver Disease Detected" if prediction else "No Liver Disease Detected"
        recs=RECOMMENDATIONS["liver"][bool(prediction)]
        st.session_state.ai_risk_scores["liver"]=round(prob*100,1)
        col1,col2=st.columns(2)
        with col1: risk_gauge(prob,"Liver Disease Risk Score")
        with col2: show_result_card(result,prob,bool(prediction))
        col3,col4=st.columns(2)
        with col3: shap_bar_chart(models["liver"],input_array,feature_names)
        with col4: show_recommendations(recs,bool(prediction))
        ap=st.session_state.active_patient
        pid=ap["id"] if ap else None
        save_screening_db(current_user["id"],pid,"Liver Disease",result,round(prob*100,1),inputs)
        st.info("💡 Risk score saved! Go to **Insurance Report** for underwriting analysis.")
        pdf_buf=generate_pdf("Liver Disease",inputs,prob,result,recs,ap)
        st.download_button("📥 Download PDF Report",pdf_buf,"liver_report.pdf","application/pdf",use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INSURANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Insurance Report":
    st.title("🛡️ Insurance Underwriting Report")
    patient_banner()
    st.divider()
    saved=st.session_state.ai_risk_scores
    if any(v>0 for v in saved.values()):
        st.success("✅ AI disease risk scores auto-imported from screenings. Review & adjust below.")
    else:
        st.info("💡 Run disease screenings first and risk scores will be auto-imported here.")

    st.subheader("👤 Health Profile")
    col1,col2,col3=st.columns(3)
    with col1:
        ins_age=st.number_input("Age (years)",min_value=18,max_value=80,value=35,key="ins_age")
        ins_bmi=st.number_input("BMI (kg/m²)",min_value=10.0,max_value=60.0,value=23.0,step=0.1,key="ins_bmi")
    with col2:
        ins_smoker=st.selectbox("Smoking Status",["Never Smoked","Former Smoker","Current Smoker"],key="ins_smoker")
        ins_alcohol=st.selectbox("Alcohol Consumption",["None","Moderate","Heavy"],key="ins_alcohol")
    with col3:
        ins_occupation=st.selectbox("Occupation Risk",["Office / Desk Job","Moderate Risk","Hazardous"],key="ins_occ")

    st.subheader("🧬 Medical Background")
    col4,col5=st.columns(2)
    with col4:
        ins_family=st.multiselect("Family History of Disease",["Heart Disease","Cancer","Diabetes","Stroke"],key="ins_family")
    with col5:
        ins_existing=st.multiselect("Existing Medical Conditions",
                                     ["Hypertension","Diabetes","Heart Disease","Cancer","Liver Disease","Kidney Disease"],key="ins_existing")

    st.subheader("🤖 AI Disease Risk Scores")
    st.caption("Auto-filled from screenings. Manually adjustable.")
    col6,col7,col8=st.columns(3)
    with col6: d_risk=st.number_input("Diabetes Risk %",0.0,100.0,value=float(saved["diabetes"]),step=0.1,key="ins_d")
    with col7: h_risk=st.number_input("Heart Disease Risk %",0.0,100.0,value=float(saved["heart"]),step=0.1,key="ins_h")
    with col8: l_risk=st.number_input("Liver Disease Risk %",0.0,100.0,value=float(saved["liver"]),step=0.1,key="ins_l")

    if any(v>0 for v in [d_risk,h_risk,l_risk]):
        gc1,gc2,gc3=st.columns(3)
        for col,label,val in [(gc1,"Diabetes",d_risk),(gc2,"Heart",h_risk),(gc3,"Liver",l_risk)]:
            with col:
                color="#EF4444" if val>=66 else "#EAB308" if val>=33 else "#22C55E"
                fig=go.Figure(go.Indicator(mode="gauge+number",value=val,
                    number={"suffix":"%","font":{"size":20,"color":color}},
                    title={"text":label,"font":{"size":11,"color":"#94A3B8"}},
                    gauge={"axis":{"range":[0,100],"tickwidth":0.5,"tickfont":{"size":8,"color":"#334155"}},
                           "bar":{"color":color,"thickness":0.25},"bgcolor":"#1E293B","borderwidth":0,
                           "steps":[{"range":[0,33],"color":"rgba(34,197,94,0.1)"},
                                    {"range":[33,66],"color":"rgba(234,179,8,0.1)"},
                                    {"range":[66,100],"color":"rgba(239,68,68,0.1)"}]}))
                fig.update_layout(height=140,margin=dict(t=28,b=0,l=10,r=10),
                                   paper_bgcolor="#0F172A",plot_bgcolor="#0F172A",font={"color":"#F1F5F9"})
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    if st.button("🔍 Generate Underwriting Report",use_container_width=True,type="primary"):
        occ_map={"Office / Desk Job":"Office","Moderate Risk":"Moderate Risk","Hazardous":"Hazardous"}
        result=compute_insurance_risk(ins_age,ins_bmi,ins_smoker,ins_alcohol,
                                       occ_map[ins_occupation],ins_family,ins_existing,d_risk,h_risk,l_risk)
        st.divider()
        st.subheader("📋 Underwriting Decision")
        dc=result["decisionColor"]
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{dc}15,{dc}05);border:1px solid {dc}50;
                    border-radius:14px;padding:28px;margin-bottom:16px;text-align:center;">
            <div style="font-size:1.8rem;font-weight:800;color:{dc};margin-bottom:10px;">{result['decision']}</div>
            <div style="display:inline-block;padding:4px 18px;background:{dc}20;border:1px solid {dc}50;
                        border-radius:20px;color:{dc};font-size:0.82rem;font-weight:700;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;">{result['category']}</div>
            <div style="color:#94A3B8;font-size:0.9rem;line-height:1.6;">{result['coverageNote']}</div>
        </div>""", unsafe_allow_html=True)

        m1,m2,m3,m4=st.columns(4)
        for col,(label,val,color) in zip([m1,m2,m3,m4],[
            ("Risk Points",str(result["riskPoints"]),"#F1F5F9"),
            ("Category",result["category"],dc),
            ("Monthly Premium",f"₹{result['finalPremium']:,}" if result["finalPremium"] else "N/A",dc),
            ("Loading Factor",
             f"+{round((result['premiumMultiplier']-1)*100)}%" if result["premiumMultiplier"] and result["premiumMultiplier"]>1
             else ("−15%" if result["premiumMultiplier"] and result["premiumMultiplier"]<1 else "N/A"),dc),
        ]):
            with col:
                st.markdown(f'<div class="ins-metric"><div class="ins-metric-label">{label}</div>'
                            f'<div class="ins-metric-value" style="color:{color};">{val}</div></div>',
                            unsafe_allow_html=True)
        st.divider()
        col_flags,col_extra=st.columns([3,2])
        with col_flags:
            st.subheader(f"⚑ Underwriting Flags ({len(result['flags'])})")
            if not result["flags"]: st.success("✅ No risk flags — Excellent health profile!")
            else:
                for sev in ["critical","high","medium","low"]:
                    for f in [x for x in result["flags"] if x["severity"]==sev]:
                        show_insurance_flag(f)
        with col_extra:
            st.subheader("📌 Underwriting Notes")
            notes=[]
            if result["riskPoints"]>110: notes.append(("🔴","Manual underwriter review mandatory before issuance."))
            if ins_existing: notes.append(("⚠️","Pre-existing conditions excluded from initial coverage."))
            if ins_smoker=="Current Smoker": notes.append(("🚬","Smoker surcharge applied. Reassess after 2 yr cessation."))
            if result["riskPoints"]<=20: notes.append(("⭐","Preferred client — eligible for loyalty discounts."))
            if not notes: notes.append(("✅","Standard terms apply. Annual reassessment recommended."))
            for icon,note in notes:
                st.markdown(f'<div class="rec-card" style="margin-bottom:8px;"><b>{icon}</b> {note}</div>',unsafe_allow_html=True)
            st.divider()
            st.subheader("📊 Policy Recommendations")
            if result["category"]=="Declined":
                recs_ins=["Reapply after 2 years with improved health metrics.",
                           "Consider accidental death benefit only policies.",
                           "Explore government health scheme options."]
            elif result["riskPoints"]>75:
                recs_ins=["Term life: reduced sum assured only.",
                           "Health cover with pre-existing disease waiting period.",
                           "Add critical illness rider after review.",
                           "Annual review with updated medical reports."]
            else:
                recs_ins=["Full term life insurance eligible.",
                           "Comprehensive health policy recommended.",
                           "Consider accident and disability riders.",
                           "Eligible for family floater plans."]
            for rec in recs_ins:
                st.markdown(f'<div class="rec-card">• {rec}</div>',unsafe_allow_html=True)

        st.divider()
        ap=st.session_state.active_patient
        patient_info={
            "Patient Name": ap["patient_name"] if ap else "N/A",
            "Patient ID":   ap["patient_id"]   if ap else "N/A",
            "Phone":        ap.get("phone","—") if ap else "—",
            "Policy No.":   ap.get("policy_number","—") if ap else "—",
            "Age":f"{ins_age} years","BMI":f"{ins_bmi:.1f} kg/m²",
            "Smoking":ins_smoker,"Alcohol":ins_alcohol,"Occupation":ins_occupation,
            "Family History":", ".join(ins_family) if ins_family else "None",
            "Existing Conditions":", ".join(ins_existing) if ins_existing else "None",
            "AI Diabetes Risk":f"{d_risk:.1f}%","AI Heart Risk":f"{h_risk:.1f}%","AI Liver Risk":f"{l_risk:.1f}%",
        }
        pid=ap["id"] if ap else None
        save_insurance_report_db(current_user["id"],pid,result,patient_info)
        st.success("✅ Report saved to patient history.")
        pdf_buf=generate_insurance_pdf(patient_info,result,result["flags"],ap)
        st.download_button("📥 Download Insurance Underwriting Report (PDF)",
                           pdf_buf,"insurance_underwriting_report.pdf","application/pdf",use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Patient History":
    st.title("📋 Patient History")
    ap=st.session_state.active_patient
    st.divider()

    view_mode = st.radio("View", ["Active Patient","All Patients"], horizontal=True, key="hist_view")
    pid_filter = (ap["id"] if ap else None) if view_mode=="Active Patient" else None

    if view_mode=="Active Patient" and not ap:
        st.warning("No active patient selected. Showing all records.")
        pid_filter=None

    tab_screenings,tab_insurance=st.tabs(["🩺 Disease Screenings","🛡️ Insurance Reports"])

    with tab_screenings:
        history=get_screenings_db(current_user["id"],pid_filter)
        if not history:
            st.info("No screenings found.")
        else:
            col1,col2,col3=st.columns(3)
            avg_risk=np.mean([h["risk_pct"] for h in history])
            col1.metric("Total Screenings",len(history))
            col2.metric("Average Risk Score",f"{avg_risk:.1f}%")
            col3.metric("High Risk Results",sum(1 for h in history if h["risk_pct"]>=66))
            df=pd.DataFrame([{
                "ID":h["id"],"Date":h["created_at"][:16],
                "Patient":h.get("patient_name","—"),"Pat.ID":h.get("pat_code","—"),
                "Disease":h["disease"],"Result":h["result"],"Risk Score":f"{h['risk_pct']}%",
            } for h in history])
            st.dataframe(df.drop(columns=["ID"]),use_container_width=True,hide_index=True)
            if len(history)>1:
                fig=px.line(x=[h["created_at"][:16] for h in history][::-1],
                             y=[h["risk_pct"] for h in history][::-1],
                             labels={"x":"Time","y":"Risk Score (%)"},markers=True,
                             color_discrete_sequence=["#3B82F6"])
                fig.add_hline(y=66,line_dash="dash",line_color="#EF4444",
                              annotation_text="High Risk",annotation_font_color="#EF4444")
                fig.add_hline(y=33,line_dash="dash",line_color="#EAB308",
                              annotation_text="Moderate Risk",annotation_font_color="#EAB308")
                fig.update_layout(paper_bgcolor="#0F172A",plot_bgcolor="#0F172A",
                                   font={"color":"#F1F5F9"},
                                   xaxis={"gridcolor":"#1E293B"},yaxis={"gridcolor":"#1E293B"},
                                   margin=dict(t=20,b=20))
                st.plotly_chart(fig,use_container_width=True)
            c1,c2=st.columns(2)
            with c1:
                del_id=st.number_input("Screening ID to delete",min_value=1,step=1,key="del_sid")
                if st.button("🗑️ Delete Screening",use_container_width=True):
                    delete_screening_db(del_id,current_user["id"])
                    st.success(f"Screening #{del_id} deleted.")
                    st.rerun()
            with c2:
                csv=df.drop(columns=["ID"]).to_csv(index=False).encode("utf-8")
                st.download_button("📥 Export CSV",csv,"screenings.csv","text/csv",use_container_width=True)

    with tab_insurance:
        ins_reports=get_insurance_reports_db(current_user["id"],pid_filter)
        if not ins_reports:
            st.info("No insurance reports found.")
        else:
            ins_df=pd.DataFrame([{
                "Date":r["created_at"][:16],
                "Patient":r.get("patient_name","—"),"Pat.ID":r.get("pat_code","—"),
                "Decision":r["decision"],"Category":r["category"],
                "Risk Points":r["risk_points"],
                "Monthly Premium":f"₹{int(r['premium_monthly']):,}" if r["premium_monthly"] else "N/A",
                "Loading":f"{r['loading_factor']:.2f}x" if r["loading_factor"] else "N/A",
            } for r in ins_reports])
            st.dataframe(ins_df,use_container_width=True,hide_index=True)
            if len(ins_reports)>1:
                fig2=px.line(x=[r["created_at"][:16] for r in ins_reports][::-1],
                              y=[r["risk_points"] for r in ins_reports][::-1],
                              labels={"x":"Date","y":"Risk Points"},markers=True,
                              color_discrete_sequence=["#8B5CF6"])
                fig2.update_layout(paper_bgcolor="#0F172A",plot_bgcolor="#0F172A",
                                    font={"color":"#F1F5F9"},
                                    xaxis={"gridcolor":"#1E293B"},yaxis={"gridcolor":"#1E293B"},
                                    margin=dict(t=20,b=20))
                st.plotly_chart(fig2,use_container_width=True)
            ins_csv=ins_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export CSV",ins_csv,"insurance_reports.csv","text/csv",use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MY ACCOUNT
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "My Account":
    st.title("👤 My Account")
    st.divider()
    col1,col2=st.columns(2)
    with col1:
        st.subheader("📋 Employee Profile")
        st.markdown(f"""
        <div style="background:rgba(30,41,59,0.8);border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px;padding:24px;">
            {"".join(f'<div style="margin-bottom:14px;"><span style="color:#475569;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">{lbl}</span><br><span style="color:#F1F5F9;font-size:1rem;font-weight:600;">{val}</span></div>'
             for lbl,val in [
                ("Full Name",    current_user.get("full_name","—")),
                ("Username",     f"@{current_user['username']}"),
                ("Email",        current_user.get("email","—")),
                ("Employee ID",  current_user.get("employee_id","—")),
                ("Department",   current_user.get("department","—")),
                ("Company",      current_user.get("company","—")),
                ("Member Since", current_user.get("created_at","—")[:10]),
             ])}
        </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Account Statistics")
        history=get_screenings_db(current_user["id"])
        ins_reps=get_insurance_reports_db(current_user["id"])
        patients=get_patients(current_user["id"])
        avg_risk=np.mean([h["risk_pct"] for h in history]) if history else 0
        s1,s2=st.columns(2)
        s1.metric("Total Patients",len(patients))
        s2.metric("Insurance Reports",len(ins_reps))
        s3,s4=st.columns(2)
        s3.metric("Total Screenings",len(history))
        s4.metric("High Risk Alerts",sum(1 for h in history if h["risk_pct"]>=66))

        st.subheader("🔑 Change Password")
        old_p=st.text_input("Current Password",type="password",key="chg_old")
        new_p=st.text_input("New Password",type="password",key="chg_new")
        new_p2=st.text_input("Confirm New Password",type="password",key="chg_new2")
        if st.button("Update Password",use_container_width=True):
            if not verify_user(current_user["username"],old_p):
                st.error("Current password incorrect.")
            elif len(new_p)<6:
                st.error("Password must be at least 6 characters.")
            elif new_p!=new_p2:
                st.error("Passwords do not match.")
            else:
                conn=get_db()
                conn.execute("UPDATE users SET password_hash=? WHERE id=?",(hash_password(new_p),current_user["id"]))
                conn.commit(); conn.close()
                st.success("✅ Password updated successfully!")