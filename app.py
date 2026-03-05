import warnings
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import shap
import io
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

# ── Minimal CSS: only for custom HTML cards, everything else is Streamlit native ──
st.markdown("""
<style>
    /* Range indicator pills */
    .ri-normal  { background:rgba(34,197,94,0.15);  border-left:3px solid #22C55E; border-radius:6px; padding:8px 14px; margin:4px 0; color:#86EFAC; font-size:0.85rem; font-weight:500; }
    .ri-warning { background:rgba(234,179,8,0.15);  border-left:3px solid #EAB308; border-radius:6px; padding:8px 14px; margin:4px 0; color:#FDE047; font-size:0.85rem; font-weight:500; }
    .ri-danger  { background:rgba(239,68,68,0.15);  border-left:3px solid #EF4444; border-radius:6px; padding:8px 14px; margin:4px 0; color:#FCA5A5; font-size:0.85rem; font-weight:500; }

    /* Result cards */
    .result-positive { background:rgba(239,68,68,0.1);  border:1px solid rgba(239,68,68,0.4);  border-radius:12px; padding:20px 24px; margin:12px 0; }
    .result-negative { background:rgba(34,197,94,0.1);  border:1px solid rgba(34,197,94,0.4);  border-radius:12px; padding:20px 24px; margin:12px 0; }
    .result-title-pos { font-size:1.2rem; font-weight:700; color:#FCA5A5; margin-bottom:6px; }
    .result-title-neg { font-size:1.2rem; font-weight:700; color:#86EFAC; margin-bottom:6px; }
    .result-sub { font-size:0.84rem; color:#94A3B8; margin-top:8px; line-height:1.5; }

    /* Recommendation cards */
    .rec-card        { background:rgba(59,130,246,0.1);  border-left:3px solid #3B82F6; border-radius:6px; padding:10px 14px; margin:5px 0; color:#CBD5E1; font-size:0.87rem; line-height:1.5; }
    .rec-card-urgent { background:rgba(249,115,22,0.1);  border-left:3px solid #F97316; color:#FED7AA; }

    /* Disease prediction card */
    .disease-card { background:rgba(30,41,59,0.8); border:1px solid rgba(59,130,246,0.3); border-radius:12px; padding:24px; margin-top:8px; }
    .disease-name { font-size:1.6rem; font-weight:700; color:#F1F5F9; margin:8px 0; }
    .disease-label-pos { background:rgba(239,68,68,0.2); color:#FCA5A5; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; display:inline-block; }
    .disease-label-med { background:rgba(234,179,8,0.2);  color:#FDE047; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; display:inline-block; }
    .disease-label-neg { background:rgba(34,197,94,0.2);  color:#86EFAC; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; display:inline-block; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "diabetes":      joblib.load("models/diabetes_model.sav"),
        "heart":         joblib.load("models/heart_disease_model.sav"),
        "liver":         joblib.load("models/liver_model.sav"),
        "breast_cancer": joblib.load("models/breast_cancer.sav"),
    }

models = load_models()

if "history" not in st.session_state:
    st.session_state.history = []


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def risk_gauge(probability: float, title: str = "Risk Score"):
    color = "#EF4444" if probability >= 0.66 else "#EAB308" if probability >= 0.33 else "#22C55E"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        title={"text": title, "font": {"size": 13, "color": "#94A3B8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                     "tickfont": {"color": "#64748B", "size": 11}},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "#1E293B",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33],   "color": "rgba(34,197,94,0.12)"},
                {"range": [33, 66],  "color": "rgba(234,179,8,0.12)"},
                {"range": [66, 100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": probability * 100},
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=30, b=0, l=20, r=20),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font={"color": "#F1F5F9"},
    )
    st.plotly_chart(fig, use_container_width=True)


def risk_badge_html(prob):
    if prob >= 0.66:
        return f'<span class="disease-label-pos">🔴 High Risk</span>', "#FCA5A5"
    elif prob >= 0.33:
        return f'<span class="disease-label-med">🟡 Moderate Risk</span>', "#FDE047"
    else:
        return f'<span class="disease-label-neg">🟢 Low Risk</span>', "#86EFAC"


def range_indicator(label, value, low, high, unit=""):
    if value == 0:
        return
    if value < low:
        cls, icon, note = "ri-warning", "⬇", f"Below normal ({low}–{high} {unit})"
    elif value > high:
        cls, icon, note = "ri-danger", "⬆", f"Above normal ({low}–{high} {unit})"
    else:
        cls, icon, note = "ri-normal", "✔", f"Within normal range ({low}–{high} {unit})"
    st.markdown(
        f'<div class="{cls}"><b>{icon} {label}:</b> {value} {unit} · {note}</div>',
        unsafe_allow_html=True
    )


def shap_bar_chart(model, input_array, feature_names):
    try:
        explainer = shap.TreeExplainer(model)
        sv_raw = explainer.shap_values(input_array)
        sv = sv_raw[1][0] if isinstance(sv_raw, list) else sv_raw[0]
        df = pd.DataFrame({"Feature": feature_names, "SHAP": sv, "Abs": np.abs(sv)})
        df = df.sort_values("Abs", ascending=True).tail(10)
        colors = ["#EF4444" if v > 0 else "#22C55E" for v in df["SHAP"]]
        fig = go.Figure(go.Bar(
            x=df["SHAP"], y=df["Feature"], orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in df["SHAP"]], textposition="outside",
            textfont={"size": 11, "color": "#CBD5E1"},
        ))
        fig.update_layout(
            title={"text": "Feature Contributions (SHAP)",
                   "font": {"size": 13, "color": "#94A3B8"}},
            xaxis={"title": "Impact on Prediction", "titlefont": {"size": 11, "color": "#64748B"},
                   "tickfont": {"size": 10, "color": "#64748B"}, "gridcolor": "#1E293B"},
            yaxis={"tickfont": {"size": 11, "color": "#CBD5E1"}, "gridcolor": "#1E293B"},
            height=340, margin=dict(l=10, r=60, t=40, b=10),
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")


def show_result_card(result_text, prob, positive):
    badge_html, score_color = risk_badge_html(prob)
    card_class = "result-positive" if positive else "result-negative"
    title_class = "result-title-pos" if positive else "result-title-neg"
    emoji = "⚠️" if positive else "✅"
    note = ("Please consult a qualified healthcare professional for a proper diagnosis."
            if positive else "Continue regular health check-ups and maintain a healthy lifestyle.")
    st.markdown(f"""
    <div class="{card_class}">
        <div class="{title_class}">{emoji} {result_text}</div>
        <div style="margin-top:8px; display:flex; align-items:center; gap:12px;">
            {badge_html}
            <span style="font-size:0.88rem; color:#94A3B8;">
                Risk Score: <b style="color:{score_color}">{prob*100:.1f}%</b>
            </span>
        </div>
        <div class="result-sub">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def show_recommendations(recs, positive):
    st.subheader("💡 Recommendations")
    for i, rec in enumerate(recs):
        cls = "rec-card rec-card-urgent" if (positive and i == 0) else "rec-card"
        prefix = "🚨 " if (positive and i == 0) else "• "
        st.markdown(f'<div class="{cls}">{prefix}{rec}</div>', unsafe_allow_html=True)


def save_to_history(disease, inputs, probability, result):
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "disease": disease, "result": result,
        "risk_pct": round(probability * 100, 1), "inputs": inputs,
    })


def generate_pdf(disease, inputs, probability, result, recommendations):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Header
    header_style = ParagraphStyle("header", fontSize=18, fontName="Helvetica-Bold",
                                   textColor=colors.white, backColor=colors.HexColor("#0F172A"),
                                   alignment=TA_CENTER, spaceAfter=4, leading=28,
                                   leftIndent=-1*cm, rightIndent=-1*cm, borderPadding=12)
    story.append(Paragraph("MediPredict AI - Health Report", header_style))
    sub_style = ParagraphStyle("sub", fontSize=9, fontName="Helvetica",
                                textColor=colors.HexColor("#94A3B8"), alignment=TA_CENTER, spaceAfter=16)
    from datetime import datetime as _dt
    story.append(Paragraph(f"Generated: {_dt.now().strftime('%B %d, %Y at %H:%M')}", sub_style))

    # Result section
    risk_color = colors.HexColor("#EF4444") if probability >= 0.66 else                  colors.HexColor("#EAB308") if probability >= 0.33 else                  colors.HexColor("#22C55E")
    label_style = ParagraphStyle("label", fontSize=11, fontName="Helvetica-Bold",
                                  textColor=colors.HexColor("#0F172A"), spaceAfter=4)
    result_style = ParagraphStyle("result", fontSize=12, fontName="Helvetica-Bold",
                                   textColor=risk_color, spaceAfter=12)
    story.append(Paragraph(f"Disease Screened: {disease}", label_style))
    story.append(Paragraph(f"Result: {result}   |   Risk Score: {probability*100:.1f}%", result_style))

    # Inputs table
    story.append(Paragraph("Input Parameters:", ParagraphStyle("h2", fontSize=11,
                            fontName="Helvetica-Bold", textColor=colors.HexColor("#0F172A"),
                            spaceAfter=6, spaceBefore=8)))
    table_data = [[str(k), str(v)] for k, v in inputs.items()]
    t = Table(table_data, colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("TEXTCOLOR",   (0,0), (0,-1), colors.HexColor("#334155")),
        ("TEXTCOLOR",   (1,0), (1,-1), colors.HexColor("#475569")),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    story.append(t)

    # Recommendations
    if recommendations:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Recommendations:", ParagraphStyle("h2", fontSize=11,
                                fontName="Helvetica-Bold", textColor=colors.HexColor("#0F172A"),
                                spaceAfter=6)))
        rec_style = ParagraphStyle("rec", fontSize=9, fontName="Helvetica",
                                    textColor=colors.HexColor("#334155"),
                                    leftIndent=10, spaceAfter=3)
        for i, rec in enumerate(recommendations):
            prefix = ">> " if i == 0 else "-  "
            story.append(Paragraph(f"{prefix}{rec}", rec_style))

    # Disclaimer
    story.append(Spacer(1, 16))
    disc_style = ParagraphStyle("disc", fontSize=7.5, fontName="Helvetica-Oblique",
                                 textColor=colors.HexColor("#94A3B8"),
                                 borderColor=colors.HexColor("#E2E8F0"),
                                 borderWidth=1, borderPadding=8, spaceAfter=0)
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI model for informational purposes only "
        "and does NOT constitute a medical diagnosis. Please consult a qualified healthcare professional.",
        disc_style))

    doc.build(story)
    buf.seek(0)
    return buf



RECOMMENDATIONS = {
    "diabetes": {
        True:  ["Consult an endocrinologist immediately.",
                "Monitor blood glucose daily; target fasting glucose < 100 mg/dL.",
                "Adopt a low-glycemic diet: whole grains, legumes, leafy greens.",
                "Aim for 150 min/week of moderate aerobic exercise.",
                "Reduce sugary beverages and refined carbohydrates.",
                "Maintain BMI between 18.5 and 24.9."],
        False: ["Maintain a balanced diet rich in fibre and low in sugar.",
                "Exercise regularly to keep insulin sensitivity high.",
                "Monitor glucose annually if you have a family history.",
                "Stay hydrated and limit alcohol intake."]
    },
    "heart": {
        True:  ["Seek immediate cardiac evaluation from a cardiologist.",
                "Monitor blood pressure; target < 120/80 mmHg.",
                "Follow a heart-healthy diet: reduce saturated fats, salt, red meat.",
                "Quit smoking — it is the #1 modifiable risk factor.",
                "Take prescribed medications consistently.",
                "Limit alcohol to 1-2 drinks per day maximum."],
        False: ["Keep cholesterol in check (LDL < 100 mg/dL).",
                "Exercise for at least 30 minutes most days.",
                "Manage stress through mindfulness or yoga.",
                "Get regular blood pressure and cholesterol screenings."]
    },
    "liver": {
        True:  ["Consult a hepatologist or gastroenterologist.",
                "Avoid alcohol completely.",
                "Follow a low-fat, high-fibre diet.",
                "Stay well-hydrated.",
                "Avoid excess OTC medications that burden the liver (e.g., acetaminophen).",
                "Get vaccinated against Hepatitis A and B."],
        False: ["Limit alcohol consumption.",
                "Maintain a healthy weight to prevent fatty liver.",
                "Eat plenty of antioxidant-rich foods.",
                "Get annual liver function tests if you have risk factors."]
    },
    "breast_cancer": {
        True:  ["Schedule an urgent appointment with an oncologist.",
                "Request a confirmatory biopsy — do not rely solely on this prediction.",
                "Discuss staging and treatment options with your medical team.",
                "Consider genetic testing (BRCA1/BRCA2) if family history exists.",
                "Seek emotional support through counselling or support groups."],
        False: ["Perform monthly self-breast exams.",
                "Schedule annual mammograms (especially over age 40).",
                "Maintain a healthy weight and limit alcohol.",
                "Discuss risk factors with your gynaecologist."]
    },
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediPredict AI")
    st.caption("AI-Powered Health Screening")
    st.divider()

    selected = option_menu(
        menu_title=None,
        options=["General Disease", "Diabetes", "Heart Disease",
                 "Liver Disease", "Breast Cancer", "Patient History"],
        icons=["activity", "droplet", "heart", "person", "gender-female", "clock-history"],
        default_index=0,
        styles={
            "container":      {"padding": "0", "background-color": "transparent"},
            "icon":           {"color": "#94A3B8", "font-size": "15px"},
            "nav-link":       {"font-size": "14px", "color": "#CBD5E1",
                               "padding": "10px 16px", "border-radius": "8px",
                               "--hover-color": "rgba(59,130,246,0.15)"},
            "nav-link-selected": {"background-color": "rgba(59,130,246,0.25)",
                                  "color": "#93C5FD", "font-weight": "600"},
        }
    )

    st.divider()
    st.warning("**⚠️ Disclaimer**  \nFor informational purposes only. Does not replace professional medical advice.")


# ══════════════════════════════════════════════════════════════════════════════
# 1. GENERAL DISEASE
# ══════════════════════════════════════════════════════════════════════════════
if selected == "General Disease":
    st.title("🔬 General Disease Prediction")
    st.caption("Select your symptoms and let the AI identify the most likely condition.")
    st.divider()

    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')

    symptoms = st.multiselect(
        "Select all symptoms you are experiencing:",
        options=disease_model.all_symptoms,
        placeholder="Type to search symptoms..."
    )
    X = prepare_symptoms_array(symptoms)

    if st.button("🔍 Run Prediction", use_container_width=True):
        if not symptoms:
            st.warning("Please select at least one symptom to continue.")
        else:
            prediction, prob = disease_model.predict(X)
            badge_html, score_color = risk_badge_html(prob)

            col1, col2 = st.columns(2)
            with col1:
                risk_gauge(prob, "Confidence Score")
            with col2:
                st.markdown(f"""
                <div class="disease-card">
                    <div style="color:#64748B; font-size:0.78rem; font-weight:600;
                                text-transform:uppercase; letter-spacing:0.5px;">Predicted Condition</div>
                    <div class="disease-name">{prediction}</div>
                    {badge_html}
                    <div style="margin-top:14px; color:#64748B; font-size:0.85rem;">
                        Confidence: <b style="color:{score_color}">{prob*100:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            tab1, tab2 = st.tabs(["📄 Description", "🛡️ Precautions"])
            with tab1:
                st.info(disease_model.describe_predicted_disease())
            with tab2:
                precautions = disease_model.predicted_disease_precautions()
                for i in range(4):
                    st.markdown(f'<div class="rec-card"><b>{i+1}.</b> {precautions[i]}</div>',
                                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DIABETES
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Diabetes":
    st.title("💉 Diabetes Risk Assessment")
    st.caption("Enter your health metrics for an AI-powered diabetes risk screening.")
    st.divider()

    with st.expander("📊 Normal Reference Ranges"):
        c1, c2 = st.columns(2)
        c1.markdown("**Glucose (fasting):** 70–99 mg/dL  \n**Blood Pressure:** 60–80 mmHg")
        c2.markdown("**BMI:** 18.5–24.9 kg/m²  \n**Insulin:** 2–25 μIU/mL")

    st.subheader("📝 Patient Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0)
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
    with col2:
        Glucose = st.number_input("Glucose (mg/dL)", min_value=0)
        Insulin = st.number_input("Insulin (μIU/mL)", min_value=0)
        Age = st.number_input("Age (years)", min_value=0)
    with col3:
        BloodPressure = st.number_input("Blood Pressure (mmHg)", min_value=0)
        BMI = st.number_input("BMI (kg/m²)", min_value=0.0, step=0.1)

    st.subheader("🩺 Input Validation")
    c1, c2 = st.columns(2)
    with c1:
        range_indicator("Glucose", Glucose, 70, 99, "mg/dL")
        range_indicator("Blood Pressure", BloodPressure, 60, 80, "mmHg")
    with c2:
        range_indicator("BMI", BMI, 18.5, 24.9, "kg/m²")
        range_indicator("Insulin", Insulin, 2, 25, "μIU/mL")

    input_array = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                              Insulin, BMI, DiabetesPedigreeFunction, Age]])
    feature_names = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                     "Insulin", "BMI", "Diabetes Pedigree", "Age"]

    st.divider()
    if st.button("🔍 Run Diabetes Screening", use_container_width=True):
        prob = models["diabetes"].predict_proba(input_array)[0][1]
        prediction = int(prob >= 0.5)
        result = "Diabetic" if prediction else "Not Diabetic"
        recs = RECOMMENDATIONS["diabetes"][bool(prediction)]
        inputs = dict(zip(feature_names, input_array[0].tolist()))

        col1, col2 = st.columns(2)
        with col1:
            risk_gauge(prob, "Diabetes Risk Score")
        with col2:
            show_result_card(result, prob, bool(prediction))

        col3, col4 = st.columns(2)
        with col3:
            shap_bar_chart(models["diabetes"], input_array, feature_names)
        with col4:
            show_recommendations(recs, bool(prediction))

        save_to_history("Diabetes", inputs, prob, result)
        pdf_buf = generate_pdf("Diabetes", inputs, prob, result, recs)
        st.download_button("📥 Download PDF Report", pdf_buf, "diabetes_report.pdf",
                           "application/pdf", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. HEART DISEASE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Heart Disease":
    st.title("❤️ Heart Disease Risk Assessment")
    st.caption("Fill in your cardiac health parameters for an AI risk evaluation.")
    st.divider()

    with st.expander("📊 Normal Reference Ranges"):
        c1, c2 = st.columns(2)
        c1.markdown("**Blood Pressure:** 90–120 mmHg  \n**Cholesterol:** < 200 mg/dL")
        c2.markdown("**Max Heart Rate:** 100–170 bpm  \n**ST Depression:** 0–1")

    st.subheader("📝 Patient Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=0)
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=0)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0)
        ca = st.number_input("Major Vessels (0–3)", min_value=0, max_value=3)
    with col2:
        sex_opts = ("Male", "Female")
        sex_val = st.selectbox("Gender", [0, 1], format_func=lambda x: sex_opts[x])
        sex = 1 if sex_val == 0 else 0
        chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0)
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, step=0.1)
        thal_opts = ("Normal", "Fixed Defect", "Reversible Defect")
        thal = st.selectbox("Thalassemia", [0, 1, 2], format_func=lambda x: thal_opts[x])
    with col3:
        cp_opts = ("Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptotic")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: cp_opts[x])
        ecg_opts = ("Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy")
        restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ecg_opts[x])
        slope_opts = ("Upsloping", "Flat", "Downsloping")
        slope = st.selectbox("Peak Exercise ST Slope", [0, 1, 2], format_func=lambda x: slope_opts[x])
        exang = 1 if st.checkbox("Exercise Induced Angina") else 0
        fbs   = 1 if st.checkbox("Fasting Blood Sugar > 120 mg/dL") else 0

    st.subheader("🩺 Input Validation")
    c1, c2, c3 = st.columns(3)
    with c1: range_indicator("Blood Pressure", trestbps, 90, 120, "mmHg")
    with c2: range_indicator("Cholesterol", chol, 0, 200, "mg/dL")
    with c3: range_indicator("Max Heart Rate", thalach, 60, 100, "bpm")

    input_array = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])
    feature_names = ["age","sex","cp","trestbps","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal"]
    inputs = {"Age": age, "Sex": sex_opts[sex_val], "Chest Pain": cp_opts[cp],
              "BP": trestbps, "Cholesterol": chol, "FBS": fbs,
              "ECG": ecg_opts[restecg], "Max HR": thalach, "Ex Angina": exang,
              "Oldpeak": oldpeak, "Slope": slope_opts[slope], "Vessels": ca, "Thal": thal_opts[thal]}

    st.divider()
    if st.button("🔍 Run Heart Disease Screening", use_container_width=True):
        prob = models["heart"].predict_proba(input_array)[0][1]
        prediction = int(prob >= 0.5)
        result = "Heart Disease Detected" if prediction else "No Heart Disease Detected"
        recs = RECOMMENDATIONS["heart"][bool(prediction)]

        col1, col2 = st.columns(2)
        with col1: risk_gauge(prob, "Heart Disease Risk Score")
        with col2: show_result_card(result, prob, bool(prediction))

        col3, col4 = st.columns(2)
        with col3: shap_bar_chart(models["heart"], input_array, feature_names)
        with col4: show_recommendations(recs, bool(prediction))

        save_to_history("Heart Disease", inputs, prob, result)
        pdf_buf = generate_pdf("Heart Disease", inputs, prob, result, recs)
        st.download_button("📥 Download PDF Report", pdf_buf, "heart_report.pdf",
                           "application/pdf", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. LIVER DISEASE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Liver Disease":
    st.title("🫀 Liver Disease Risk Assessment")
    st.caption("Enter your liver function test values for an AI-powered risk evaluation.")
    st.divider()

    with st.expander("📊 Normal Reference Ranges"):
        c1, c2 = st.columns(2)
        c1.markdown("**Total Bilirubin:** 0.1–1.2 mg/dL  \n**Direct Bilirubin:** 0.0–0.3 mg/dL  \n**Alk Phosphotase:** 44–147 IU/L  \n**ALT:** 7–56 IU/L")
        c2.markdown("**AST:** 10–40 IU/L  \n**Total Proteins:** 6.3–8.2 g/dL  \n**Albumin:** 3.5–5.0 g/dL  \n**A/G Ratio:** 1.0–2.5")

    st.subheader("📝 Patient Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        sex_opts = ("Male", "Female")
        sex_val = st.selectbox("Gender", [0, 1], format_func=lambda x: sex_opts[x])
        Sex = 0 if sex_val == 0 else 1
        age = st.number_input("Age (years)", min_value=0)
        Total_Bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, step=0.1)
        Direct_Bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, step=0.1)
    with col2:
        Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase (IU/L)", min_value=0)
        Alamine_Aminotransferase = st.number_input("ALT (IU/L)", min_value=0)
        Aspartate_Aminotransferase = st.number_input("AST (IU/L)", min_value=0)
    with col3:
        Total_Protiens = st.number_input("Total Proteins (g/dL)", min_value=0.0, step=0.1)
        Albumin = st.number_input("Albumin (g/dL)", min_value=0.0, step=0.1)
        Albumin_and_Globulin_Ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0, step=0.01)

    st.subheader("🩺 Input Validation")
    c1, c2 = st.columns(2)
    with c1:
        range_indicator("Total Bilirubin", Total_Bilirubin, 0.1, 1.2, "mg/dL")
        range_indicator("Direct Bilirubin", Direct_Bilirubin, 0.0, 0.3, "mg/dL")
        range_indicator("Alk Phosphotase", Alkaline_Phosphotase, 44, 147, "IU/L")
    with c2:
        range_indicator("ALT", Alamine_Aminotransferase, 7, 56, "IU/L")
        range_indicator("AST", Aspartate_Aminotransferase, 10, 40, "IU/L")
        range_indicator("Albumin", Albumin, 3.5, 5.0, "g/dL")

    input_array = np.array([[Sex, age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                              Alamine_Aminotransferase, Aspartate_Aminotransferase,
                              Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]])
    feature_names = ["Sex","Age","Total Bilirubin","Direct Bilirubin","Alk Phosphotase",
                     "ALT","AST","Total Proteins","Albumin","A/G Ratio"]
    inputs = dict(zip(feature_names, [sex_opts[sex_val], age, Total_Bilirubin, Direct_Bilirubin,
                                       Alkaline_Phosphotase, Alamine_Aminotransferase,
                                       Aspartate_Aminotransferase, Total_Protiens,
                                       Albumin, Albumin_and_Globulin_Ratio]))

    st.divider()
    if st.button("🔍 Run Liver Disease Screening", use_container_width=True):
        prob = models["liver"].predict_proba(input_array)[0][1]
        prediction = int(prob >= 0.5)
        result = "Liver Disease Detected" if prediction else "No Liver Disease Detected"
        recs = RECOMMENDATIONS["liver"][bool(prediction)]

        col1, col2 = st.columns(2)
        with col1: risk_gauge(prob, "Liver Disease Risk Score")
        with col2: show_result_card(result, prob, bool(prediction))

        col3, col4 = st.columns(2)
        with col3: shap_bar_chart(models["liver"], input_array, feature_names)
        with col4: show_recommendations(recs, bool(prediction))

        save_to_history("Liver Disease", inputs, prob, result)
        pdf_buf = generate_pdf("Liver Disease", inputs, prob, result, recs)
        st.download_button("📥 Download PDF Report", pdf_buf, "liver_report.pdf",
                           "application/pdf", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. BREAST CANCER
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Breast Cancer":
    st.title("🎗️ Breast Cancer Risk Assessment")
    st.caption("Adjust the tumour cell nucleus measurements for an AI malignancy risk evaluation.")
    st.divider()

    st.subheader("📝 Tumour Measurements")
    tab_mean, tab_se, tab_worst = st.tabs(["📐 Mean Values", "📏 Standard Error", "📍 Worst Values"])

    with tab_mean:
        col1, col2, col3 = st.columns(3)
        with col1:
            radius_mean            = st.slider("Radius Mean", 6.0, 30.0, 15.0)
            texture_mean           = st.slider("Texture Mean", 9.0, 40.0, 20.0)
            perimeter_mean         = st.slider("Perimeter Mean", 43.0, 190.0, 90.0)
            area_mean              = st.slider("Area Mean", 143.0, 2501.0, 750.0)
        with col2:
            smoothness_mean        = st.slider("Smoothness Mean", 0.05, 0.25, 0.1)
            compactness_mean       = st.slider("Compactness Mean", 0.02, 0.3, 0.15)
            concavity_mean         = st.slider("Concavity Mean", 0.0, 0.5, 0.2)
        with col3:
            concave_points_mean    = st.slider("Concave Points Mean", 0.0, 0.2, 0.1)
            symmetry_mean          = st.slider("Symmetry Mean", 0.1, 1.0, 0.5)
            fractal_dimension_mean = st.slider("Fractal Dimension Mean", 0.01, 0.1, 0.05)

    with tab_se:
        col1, col2, col3 = st.columns(3)
        with col1:
            radius_se              = st.slider("Radius SE", 0.1, 3.0, 1.0)
            texture_se             = st.slider("Texture SE", 0.2, 2.0, 1.0)
            perimeter_se           = st.slider("Perimeter SE", 1.0, 30.0, 10.0)
            area_se                = st.slider("Area SE", 6.0, 500.0, 150.0)
        with col2:
            smoothness_se          = st.slider("Smoothness SE", 0.001, 0.03, 0.01)
            compactness_se         = st.slider("Compactness SE", 0.002, 0.2, 0.1)
            concavity_se           = st.slider("Concavity SE", 0.0, 0.05, 0.02)
        with col3:
            concave_points_se      = st.slider("Concave Points SE", 0.0, 0.03, 0.01)
            symmetry_se            = st.slider("Symmetry SE", 0.1, 1.0, 0.5)
            fractal_dimension_se   = st.slider("Fractal Dimension SE", 0.01, 0.1, 0.05)

    with tab_worst:
        col1, col2, col3 = st.columns(3)
        with col1:
            radius_worst            = st.slider("Radius Worst", 7.0, 40.0, 20.0)
            texture_worst           = st.slider("Texture Worst", 12.0, 50.0, 25.0)
            perimeter_worst         = st.slider("Perimeter Worst", 50.0, 250.0, 120.0)
            area_worst              = st.slider("Area Worst", 185.0, 4250.0, 1500.0)
        with col2:
            smoothness_worst        = st.slider("Smoothness Worst", 0.07, 0.3, 0.15)
            compactness_worst       = st.slider("Compactness Worst", 0.03, 0.6, 0.3)
            concavity_worst         = st.slider("Concavity Worst", 0.0, 0.8, 0.4)
        with col3:
            concave_points_worst    = st.slider("Concave Points Worst", 0.0, 0.2, 0.1)
            symmetry_worst          = st.slider("Symmetry Worst", 0.1, 1.0, 0.5)
            fractal_dimension_worst = st.slider("Fractal Dimension Worst", 0.01, 0.2, 0.1)

    feature_cols = [
        'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
        'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
        'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
        'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
        'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
        'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
    ]
    values = [
        radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,
        compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,
        radius_se,texture_se,perimeter_se,area_se,smoothness_se,
        compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,
        radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,
        compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst
    ]
    user_input = pd.DataFrame([values], columns=feature_cols)

    st.divider()
    if st.button("🔍 Predict Breast Cancer Risk", use_container_width=True):
        prob = models["breast_cancer"].predict_proba(user_input)[0][1]
        prediction = int(prob >= 0.5)
        result = "Malignant — Cancer Indicators Detected" if prediction else "Benign — No Cancer Indicators"
        recs = RECOMMENDATIONS["breast_cancer"][bool(prediction)]

        col1, col2 = st.columns(2)
        with col1: risk_gauge(prob, "Malignancy Risk Score")
        with col2: show_result_card(result, prob, bool(prediction))

        col3, col4 = st.columns(2)
        with col3: shap_bar_chart(models["breast_cancer"], user_input.values, feature_cols)
        with col4: show_recommendations(recs, bool(prediction))

        save_to_history("Breast Cancer", dict(zip(feature_cols, values)), prob, result)
        pdf_buf = generate_pdf("Breast Cancer", dict(zip(feature_cols, values)), prob, result, recs)
        st.download_button("📥 Download PDF Report", pdf_buf, "breast_cancer_report.pdf",
                           "application/pdf", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. PATIENT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Patient History":
    st.title("📋 Patient Screening History")
    st.caption("All predictions made in this session, with trend analysis and export.")
    st.divider()

    if not st.session_state.history:
        st.info("No screenings completed yet. Run a disease screening to see your history here.")
    else:
        col1, col2, col3 = st.columns(3)
        total     = len(st.session_state.history)
        avg_risk  = np.mean([h["risk_pct"] for h in st.session_state.history])
        high_risk = sum(1 for h in st.session_state.history if h["risk_pct"] >= 66)
        col1.metric("Total Screenings", total)
        col2.metric("Average Risk Score", f"{avg_risk:.1f}%")
        col3.metric("High Risk Results", high_risk)

        st.subheader("📄 Screening Log")
        df = pd.DataFrame([{
            "Date / Time": h["timestamp"], "Disease": h["disease"],
            "Result": h["result"], "Risk Score": f"{h['risk_pct']}%",
        } for h in st.session_state.history])
        st.dataframe(df, use_container_width=True, hide_index=True)

        if len(st.session_state.history) > 1:
            st.subheader("📈 Risk Score Trend")
            fig = px.line(
                x=[h["timestamp"] for h in st.session_state.history],
                y=[h["risk_pct"] for h in st.session_state.history],
                labels={"x": "Time", "y": "Risk Score (%)"},
                markers=True,
                color_discrete_sequence=["#3B82F6"],
            )
            fig.add_hline(y=66, line_dash="dash", line_color="#EF4444",
                          annotation_text="High Risk", annotation_font_color="#EF4444")
            fig.add_hline(y=33, line_dash="dash", line_color="#EAB308",
                          annotation_text="Moderate Risk", annotation_font_color="#EAB308")
            fig.update_layout(
                paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                font={"color": "#F1F5F9"},
                xaxis={"gridcolor": "#1E293B"}, yaxis={"gridcolor": "#1E293B"},
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export as CSV", csv, "screening_history.csv",
                               "text/csv", use_container_width=True)
        with c2:
            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state.history = []
                st.rerun()