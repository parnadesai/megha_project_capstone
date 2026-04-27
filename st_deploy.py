import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, log_loss
import warnings
warnings.filterwarnings('ignore')
 
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
 
/* Dark medical theme */
.stApp {
    background-color: #0b0f1a;
    color: #e8eaf2;
}
 
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
 
/* Hero header */
.hero {
    background: linear-gradient(135deg, #1a1f35 0%, #0f1829 50%, #12172d 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(220,50,50,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #8892b0;
    font-size: 1.05rem;
    margin: 0;
    font-weight: 300;
}
.hero .accent { color: #e05555; }
 
/* Section titles */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #cdd6f4;
    margin: 0 0 1rem 0;
    border-left: 3px solid #e05555;
    padding-left: 0.75rem;
}
 
/* Metric cards */
.metric-card {
    background: #141929;
    border: 1px solid #1e2a42;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #64b5f6;
    display: block;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1px;
}
 
/* Risk result box */
.risk-high {
    background: linear-gradient(135deg, #3d1515 0%, #2a0f0f 100%);
    border: 1px solid #e05555;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #0f3d20 0%, #0a2a15 100%);
    border: 1px solid #4caf50;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.risk-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    margin: 0.5rem 0;
}
 
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid #1e2a42;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #8892b0 !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
 
/* Button */
.stButton > button {
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.85;
}
 
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #141929;
    border-radius: 10px;
    padding: 0.25rem;
    gap: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8892b0;
    border-radius: 8px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e2a42 !important;
    color: #cdd6f4 !important;
}
 
/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2a42;
    border-radius: 10px;
}
 
/* Selectbox and inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background-color: #141929 !important;
    color: #e8eaf2 !important;
    border-color: #1e2a42 !important;
}
 
div[data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)
 
# ── Load & preprocess data ───────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    try:
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please place `healthcare-dataset-stroke-data.csv` in the same directory.")
        st.stop()
 
    df.drop('id', axis=1, inplace=True)
    df['bmi'].fillna(df['bmi'].mode()[0], inplace=True)
    df['ever_married'].replace({'Yes': 1, 'No': 0}, inplace=True)
    df['gender'].replace({'Male': 1, 'Female': 0, 'Other': 2}, inplace=True)
    df['Residence_type'].replace({'Urban': 1, 'Rural': 0}, inplace=True)
    df['smoking_status'].replace({'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}, inplace=True)
    df['work_type'].replace({'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Never_worked': 4}, inplace=True)
    df['age'] = pd.cut(x=df['age'], bins=[0, 12, 19, 30, 60, 100], labels=[0, 1, 2, 3, 4])
    df['age'] = df['age'].cat.codes  # safely convert Categorical to int (-1 for NaN)
    df['age'] = df['age'].replace(-1, 4)  # fallback: assign seniors if bin missed
    # Ensure every column is a plain numeric dtype sklearn can consume
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    return df
 
@st.cache_resource
def train_models(df):
    X = df.drop('stroke', axis=1).astype(float)
    y = df['stroke'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM':                 SVC(probability=True),
        'Decision Tree':       DecisionTreeClassifier(),
        'KNN':                 KNeighborsClassifier(),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        results[name] = {
            'model': model,
            'preds': preds,
            'accuracy': accuracy_score(y_test, preds),
            'f1': f1_score(y_test, preds, zero_division=0),
            'mae': mean_absolute_error(y_test, preds),
            'mse': mean_squared_error(y_test, preds),
            'log_loss': log_loss(y_test, proba) if proba is not None else None,
            'cm': metrics.confusion_matrix(y_test, preds),
        }
    return results, X_test, y_test, X.columns.tolist()
 
# ── Set matplotlib dark style ─────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#141929',
    'axes.facecolor':   '#141929',
    'axes.edgecolor':   '#1e2a42',
    'axes.labelcolor':  '#8892b0',
    'xtick.color':      '#8892b0',
    'ytick.color':      '#8892b0',
    'text.color':       '#cdd6f4',
    'grid.color':       '#1e2a42',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
})
 
# ── App ───────────────────────────────────────────────────────────────────────
df = load_and_preprocess()
 
# Hero
st.markdown("""
<div class="hero">
  <h1>🧠 Stroke <span class="accent">Risk</span> Predictor</h1>
  <p>Clinical decision support powered by machine learning — enter patient vitals in the sidebar to assess stroke likelihood.</p>
</div>
""", unsafe_allow_html=True)
 
with st.spinner("Training models on healthcare dataset…"):
    model_results, X_test, y_test, feature_cols = train_models(df)
 
# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.markdown("## Patient Profile")
st.sidebar.markdown("---")
 
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
age_raw = st.sidebar.slider("Age", 1, 100, 45)
hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt Job", "Children", "Never Worked"])
residence = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose = st.sidebar.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 106.0, step=0.5)
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 28.0, step=0.1)
smoking = st.sidebar.selectbox("Smoking Status", ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"])
model_choice = st.sidebar.selectbox("Prediction Model", list(model_results.keys()))
 
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Predict Stroke Risk")
 
# ── Encode sidebar inputs ─────────────────────────────────────────────────────
gender_enc      = {"Female": 0, "Male": 1, "Other": 2}[gender]
hypertension_enc = 1 if hypertension == "Yes" else 0
heart_disease_enc = 1 if heart_disease == "Yes" else 0
ever_married_enc = 1 if ever_married == "Yes" else 0
work_enc        = {"Private": 0, "Self-employed": 1, "Children": 2, "Govt Job": 3, "Never Worked": 4}[work_type]
residence_enc   = 1 if residence == "Urban" else 0
smoking_enc     = {"Formerly Smoked": 0, "Never Smoked": 1, "Smokes": 2, "Unknown": 3}[smoking]
age_enc         = pd.cut([age_raw], bins=[0, 12, 19, 30, 60, 100], labels=[0, 1, 2, 3, 4])[0]
age_enc         = int(age_enc) if not pd.isna(age_enc) else 4
 
patient_data = np.array([[gender_enc, age_enc, hypertension_enc, heart_disease_enc,
                          ever_married_enc, work_enc, residence_enc, avg_glucose, bmi, smoking_enc]])
 
# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 Prediction", "📊 Model Performance", "📈 Data Insights"])
 
# ── TAB 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    if predict_btn:
        chosen = model_results[model_choice]
        pred = chosen['model'].predict(patient_data)[0]
        proba = chosen['model'].predict_proba(patient_data)[0][1] * 100 if hasattr(chosen['model'], 'predict_proba') else None
 
        col1, col2 = st.columns([1.2, 1])
 
        with col1:
            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                  <div style="font-size:3rem">⚠️</div>
                  <div class="risk-title" style="color:#ff6b6b">HIGH STROKE RISK</div>
                  <p style="color:#ff9999;margin:0.5rem 0 0 0">
                    Model predicts elevated stroke likelihood for this patient.
                  </p>
                  {'<p style="font-size:1.5rem;color:#ff6b6b;font-weight:700;margin-top:0.5rem">' + f'{proba:.1f}% probability</p>' if proba is not None else ''}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                  <div style="font-size:3rem">✅</div>
                  <div class="risk-title" style="color:#69d699">LOW STROKE RISK</div>
                  <p style="color:#a8e6bf;margin:0.5rem 0 0 0">
                    No significant stroke indicators detected for this patient.
                  </p>
                  {'<p style="font-size:1.5rem;color:#69d699;font-weight:700;margin-top:0.5rem">' + f'{proba:.1f}% probability</p>' if proba is not None else ''}
                </div>""", unsafe_allow_html=True)
 
            st.markdown(f"""
            <br>
            <div class="metric-card">
              <span class="metric-label">Model Used</span>
              <span class="metric-value" style="font-size:1.2rem;color:#cdd6f4">{model_choice}</span>
              <br>
              <span class="metric-label">Test Accuracy</span>
              <span class="metric-value">{chosen['accuracy']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
 
        with col2:
            st.markdown('<p class="section-title">Patient Summary</p>', unsafe_allow_html=True)
            summary = pd.DataFrame({
                'Feature': ['Gender', 'Age', 'Hypertension', 'Heart Disease',
                            'Married', 'Work Type', 'Residence', 'Glucose', 'BMI', 'Smoking'],
                'Value': [gender, f"{age_raw} yrs ({['Child','Teen','Young Adult','Adult','Senior'][age_enc]})",
                          hypertension, heart_disease, ever_married, work_type,
                          residence, f"{avg_glucose} mg/dL", f"{bmi:.1f}", smoking]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#4a5568;background:#0d1220;border-radius:14px;border:1px dashed #1e2a42">
          <div style="font-size:3rem;margin-bottom:1rem">🧬</div>
          <div style="font-size:1.1rem;color:#6b7280">Enter patient details in the sidebar<br>and click <strong style="color:#8892b0">Predict Stroke Risk</strong></div>
        </div>
        """, unsafe_allow_html=True)
 
# ── TAB 2: Model Performance ──────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-title">Accuracy Comparison</p>', unsafe_allow_html=True)
 
    names = list(model_results.keys())
    accs  = [model_results[n]['accuracy'] * 100 for n in names]
    f1s   = [model_results[n]['f1'] * 100 for n in names]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    colors = ['#e05555', '#5576e0', '#55e09a', '#e09a55']
    bars = axes[0].bar(names, accs, color=colors, width=0.5, edgecolor='none', zorder=3)
    axes[0].set_ylim(80, 100)
    axes[0].set_title("Accuracy (%)", fontsize=12, pad=10)
    axes[0].grid(axis='y', zorder=0)
    axes[0].set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='#cdd6f4')
 
    bars2 = axes[1].bar(names, f1s, color=colors, width=0.5, edgecolor='none', zorder=3)
    axes[1].set_ylim(0, 60)
    axes[1].set_title("F1 Score (%)", fontsize=12, pad=10)
    axes[1].grid(axis='y', zorder=0)
    axes[1].set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars2, f1s):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='#cdd6f4')
 
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    st.markdown('<p class="section-title">Confusion Matrices</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (name, res) in enumerate(model_results.items()):
        with cols[i]:
            fig2, ax = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                        linewidths=0.5, linecolor='#0d1220',
                        cbar=False, ax=ax,
                        annot_kws={"size": 12, "color": "#ffffff"})
            ax.set_title(name, fontsize=9, color='#cdd6f4')
            ax.set_xlabel('Predicted', fontsize=8)
            ax.set_ylabel('Actual', fontsize=8)
            st.pyplot(fig2)
            plt.close()
 
    st.markdown('<p class="section-title">Detailed Metrics</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame([{
        'Model': name,
        'Accuracy': f"{res['accuracy']*100:.2f}%",
        'F1 Score': f"{res['f1']*100:.2f}%",
        'MAE': f"{res['mae']:.4f}",
        'MSE': f"{res['mse']:.4f}",
        'Log Loss': f"{res['log_loss']:.4f}" if res['log_loss'] else 'N/A',
    } for name, res in model_results.items()])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
 
# ── TAB 3: Data Insights ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)
 
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val) in zip([c1, c2, c3, c4], [
        ("Total Records", len(df)),
        ("Stroke Cases", int(df['stroke'].sum())),
        ("No Stroke", int((df['stroke'] == 0).sum())),
        ("Features", len(df.columns) - 1),
    ]):
        col.markdown(f"""
        <div class="metric-card">
          <span class="metric-value">{val:,}</span>
          <span class="metric-label">{label}</span>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Stroke by Age Group & Key Factors</p>', unsafe_allow_html=True)
 
    age_labels = {0: 'Child (0-12)', 1: 'Teen (13-19)', 2: 'Young Adult (20-30)', 3: 'Adult (31-60)', 4: 'Senior (61+)'}
    df_plot = df.copy()
    df_plot['age_label'] = df_plot['age'].map(age_labels)
 
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5))
 
    stroke_colors = {0: '#4a90d9', 1: '#e05555'}
    for ax, (col_name, title, xlabels) in zip(axes3, [
        ('age', 'Stroke by Age Group',
         ['Child', 'Teen', 'Yng Adult', 'Adult', 'Senior']),
        ('hypertension', 'Stroke by Hypertension', ['No', 'Yes']),
        ('smoking_status', 'Stroke by Smoking', ['Formerly', 'Never', 'Smokes', 'Unknown']),
    ]):
        for stroke_val, color in stroke_colors.items():
            subset = df_plot[df_plot['stroke'] == stroke_val]
            counts = subset[col_name].value_counts().sort_index()
            ax.bar([str(x) for x in counts.index], counts.values,
                   label='Stroke' if stroke_val == 1 else 'No Stroke',
                   color=color, alpha=0.85, width=0.35,
                   align='edge' if stroke_val == 1 else 'center')
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y')
        ax.set_xticklabels(xlabels, rotation=20, ha='right', fontsize=8)
 
    handles = [mpatches.Patch(color='#4a90d9', label='No Stroke'),
               mpatches.Patch(color='#e05555', label='Stroke')]
    fig3.legend(handles=handles, loc='upper right', framealpha=0.2, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
 
    st.markdown('<p class="section-title">Correlation Heatmap</p>', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, linecolor='#0d1220',
                cbar_kws={'shrink': 0.8}, ax=ax4,
                annot_kws={'size': 8})
    ax4.set_title('Feature Correlation Matrix', fontsize=12, pad=12)
    st.pyplot(fig4)
    plt.close()
 
    with st.expander("📋 View Raw Data Sample"):
        raw_df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        st.dataframe(raw_df.head(20), use_container_width=True)
