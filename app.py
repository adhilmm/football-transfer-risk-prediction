import streamlit as st
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Transfer Risk Predictor",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0f1e; color: #f0f0f0; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 60%, #0a1628 100%); }
.hero { text-align: center; padding: 2.5rem 1rem 1rem 1rem; }
.hero-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 3.5rem; letter-spacing: 4px;
    background: linear-gradient(90deg, #00e676, #1de9b6, #00b0ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; margin-bottom: 0.2rem;
}
.hero-sub { font-size: 0.9rem; color: #7ecfb3; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.4rem; }
.hero-desc { font-size: 0.88rem; color: #8899aa; max-width: 460px; margin: 0 auto 0.5rem auto; line-height: 1.6; }
.divider {
    height: 2px; background: linear-gradient(90deg, transparent, #00e676, #00b0ff, transparent);
    margin: 1rem auto 2rem auto; max-width: 380px; border-radius: 2px;
}
.section-header { font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; letter-spacing: 3px; color: #00e676; text-align: center; margin-bottom: 0.2rem; }
.section-sub { text-align: center; color: #7799aa; font-size: 0.82rem; margin-bottom: 1.8rem; letter-spacing: 1px; }
.result-high {
    background: linear-gradient(135deg, rgba(255,59,48,0.15), rgba(255,59,48,0.04));
    border: 2px solid #ff3b30; border-radius: 18px; padding: 2rem; text-align: center; margin-top: 1.5rem;
}
.result-low {
    background: linear-gradient(135deg, rgba(0,230,118,0.15), rgba(0,230,118,0.04));
    border: 2px solid #00e676; border-radius: 18px; padding: 2rem; text-align: center; margin-top: 1.5rem;
}
.result-icon { font-size: 3rem; }
.result-label { font-family: 'Bebas Neue', sans-serif; font-size: 2.8rem; letter-spacing: 5px; margin: 0.3rem 0; }
.result-label-high { color: #ff3b30; }
.result-label-low  { color: #00e676; }
.result-desc { font-size: 0.88rem; color: #aabbcc; margin-top: 0.4rem; line-height: 1.6; }
.stButton > button {
    background: linear-gradient(90deg, #00e676, #00b0ff) !important; color: #0a0f1e !important;
    font-family: 'Bebas Neue', sans-serif !important; font-size: 1.15rem !important;
    letter-spacing: 3px !important; border: none !important; border-radius: 10px !important;
    padding: 0.65rem 2rem !important; width: 100% !important; margin-top: 0.5rem !important;
}
.stButton > button:hover { opacity: 0.82 !important; }
div[data-testid="stNumberInput"] label {
    color: #9dbfb0 !important; font-size: 0.8rem !important;
    font-weight: 600 !important; letter-spacing: 0.4px !important; text-transform: uppercase !important;
}
.badge {
    display: inline-block; background: rgba(0,230,118,0.12); border: 1px solid rgba(0,230,118,0.3);
    color: #00e676; border-radius: 20px; padding: 0.2rem 0.9rem; font-size: 0.75rem;
    letter-spacing: 1.5px; font-weight: 600; text-transform: uppercase; margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "position" not in st.session_state:
    st.session_state.position = None
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models(pos):
    base = os.path.dirname(os.path.abspath(__file__))
    names = {
        "gk":  ("model_gk (1).pkl",  "imputer_gk (1).pkl",  "scaler_gk (1).pkl",  "encoder_gk (1).pkl"),
        "def": ("model_def (1).pkl", "imputer_def (1).pkl", "scaler_def (1).pkl", "encoder_def (1).pkl"),
        "mid": ("model_mid.pkl",     "imputer_mid.pkl",     "scaler_mid.pkl",     "encoder_mid.pkl"),
        "fw":  ("model_fw.pkl",      "imputer_fw.pkl",      "scaler_fw.pkl",      "encoder_fw.pkl"),
    }
    model_f, imp_f, scl_f, enc_f = names[pos]
    try:
        model   = joblib.load(os.path.join(base, model_f))
        scaler  = joblib.load(os.path.join(base, scl_f))
        encoder = joblib.load(os.path.join(base, enc_f))
        return model, scaler, encoder, None
    except FileNotFoundError as e:
        return None, None, None, str(e)

# ─── Feature Definitions ───────────────────────────────────────────────────────
# (key, label, min, max, default, step, is_float)
COMMON = [
    ("age",                      "Age",                   17,  40,  25, 1, False),
    ("overall_rating",           "Overall Rating",        60,  99,  75, 1, False),
    ("potential_rating",         "Potential Rating",      60,  99,  80, 1, False),
    ("market_value_million_eur", "Market Value (M EUR)",   1, 200,  30, 1, True),
    ("contract_years_left",      "Contract Years Left",    0,   5,   2, 1, False),
    ("injury_history",           "Injury History",         0,   8,   2, 1, False),
    ("matches_played",           "Matches Played",         5,  45,  25, 1, False),
]

POSITION_FEATURES = {
    "GK": COMMON + [
        ("saves",           "Saves",            30, 130,  75, 1, False),
        ("clean_sheets",    "Clean Sheets",      2,  21,  10, 1, False),
        ("goals_conceded",  "Goals Conceded",   10,  60,  30, 1, False),
        ("save_percentage", "Save Percentage",  40, 100,  70, 1, True),
    ],
    "DEF": COMMON + [
        ("tackles",          "Tackles",          20, 110,  65, 1, False),
        ("interceptions",    "Interceptions",    10,  90,  50, 1, False),
        ("clearances",       "Clearances",       20, 160,  90, 1, False),
        ("aerial_duels_won", "Aerial Duels Won",  2,  90,  60, 1, False),
    ],
    "MID": COMMON + [
        ("key_passes",      "Key Passes",        10, 130,  70, 1, False),
        ("dribbles",        "Dribbles",          10, 100,  55, 1, False),
        ("chances_created", "Chances Created",   10, 100,  60, 1, False),
        ("pass_accuracy",   "Pass Accuracy %",   65, 100,  80, 1, True),
    ],
    "FW": COMMON + [
        ("goals",           "Goals",              0,  35,  15, 1, False),
        ("assists",         "Assists",            0,  17,   8, 1, False),
        ("shots_on_target", "Shots on Target",    5,  90,  50, 1, False),
        ("conversion_rate", "Conversion Rate %",  0, 100,  40, 1, True),
    ],
}

POSITION_INFO = {
    "GK":  {"label": "Goalkeeper", "icon": "🧤", "pkl_key": "gk"},
    "DEF": {"label": "Defender",   "icon": "🛡️", "pkl_key": "def"},
    "MID": {"label": "Midfielder", "icon": "⚙️", "pkl_key": "mid"},
    "FW":  {"label": "Forward",    "icon": "⚡",  "pkl_key": "fw"},
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div class="hero">
        <div class="hero-sub">AI-Powered Scouting Tool</div>
        <div class="hero-title">⚽ Football Transfer<br>Risk Predictor</div>
        <div class="divider"></div>
        <div class="hero-desc">
            Enter a player's performance stats to predict whether they are at
            <strong style="color:#ff3b30">High</strong> or
            <strong style="color:#00e676">Low</strong> transfer risk.
            Select a position below to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='margin-top:1.5rem'>SELECT POSITION</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Choose the player's playing position</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, pos in zip([col1, col2, col3, col4], ["GK", "DEF", "MID", "FW"]):
        info = POSITION_INFO[pos]
        with col:
            if st.button(f"{info['icon']}  {pos}\n{info['label']}", key=f"btn_{pos}", use_container_width=True):
                st.session_state.position = pos
                st.session_state.page = "predict"
                st.rerun()

    # ── Prediction History ──
    if st.session_state.history:
        st.markdown("<div class='section-header' style='margin-top:2.5rem;font-size:1.5rem'>📜 PREDICTION HISTORY</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>All predictions made this session</div>", unsafe_allow_html=True)

        import pandas as pd
        df_hist = pd.DataFrame(st.session_state.history)
        # Style the dataframe
        def color_result(val):
            if 'HIGH' in str(val):
                return 'color: #ff3b30; font-weight: bold;'
            elif 'LOW' in str(val):
                return 'color: #00e676; font-weight: bold;'
            return ''
        styled = df_hist.style.applymap(color_result, subset=['Result'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        if st.button("🗑️  CLEAR HISTORY", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

    st.markdown("""
    <div style='text-align:center;margin-top:3rem;color:#2a3a4a;font-size:0.75rem;letter-spacing:1px;'>
        RANDOM FOREST MODEL · 4 POSITION DATASETS · HIGH / LOW RISK CLASSIFICATION
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
def page_predict():
    pos  = st.session_state.position
    info = POSITION_INFO[pos]
    feats = POSITION_FEATURES[pos]

    if st.button("← Back to Positions", key="back"):
        st.session_state.page = "home"
        st.session_state.position = None
        st.rerun()

    st.markdown(f"""
    <div class="hero" style="padding-top:0.8rem">
        <div class="hero-sub">Position Selected</div>
        <div class="hero-title">{info['icon']} {info['label']}</div>
        <div class="divider"></div>
    </div>
    <div style='text-align:center;margin-bottom:1.5rem'>
        <span class='badge'>⚽ Transfer Risk Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>PLAYER STATS</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Enter the player's statistics below</div>", unsafe_allow_html=True)

    input_values = {}
    col_a, col_b = st.columns(2)
    for i, (key, label, mn, mx, default, step, is_float) in enumerate(feats):
        with (col_a if i % 2 == 0 else col_b):
            if is_float:
                val = st.number_input(label, min_value=float(mn), max_value=float(mx),
                                      value=float(default), step=float(step), key=f"inp_{key}")
            else:
                val = st.number_input(label, min_value=int(mn), max_value=int(mx),
                                      value=int(default), step=int(step), key=f"inp_{key}")
            input_values[key] = val

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  PREDICT TRANSFER RISK", key="predict_btn"):
        pkl_key = info["pkl_key"]
        model, scaler, encoder, err = load_models(pkl_key)

        if err:
            st.error(
                f"⚠️ Model files not found. Place all .pkl files in the same folder as app.py\n\n"
                f"Missing: `{err}`"
            )
        else:
            feature_order = [f[0] for f in feats]
            X = np.array([[input_values[k] for k in feature_order]])
            X_scl = scaler.transform(X)
            pred  = model.predict(X_scl)[0]

            try:
                label_str = encoder.inverse_transform([pred])[0]
            except Exception:
                label_str = str(pred)

            is_high = "HIGH" in str(label_str).upper()

            if is_high:
                st.markdown("""
                <div class="result-high">
                    <div class="result-icon">🔴</div>
                    <div class="result-label result-label-high">HIGH RISK</div>
                    <div class="result-desc">
                        This player shows a <strong style="color:#ff3b30">HIGH transfer risk</strong>.<br>
                        They are likely to be transferred or leave the club soon.<br>
                        Consider contract renewal or replacement planning.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-low">
                    <div class="result-icon">🟢</div>
                    <div class="result-label result-label-low">LOW RISK</div>
                    <div class="result-desc">
                        This player shows a <strong style="color:#00e676">LOW transfer risk</strong>.<br>
                        They are likely to remain at their current club.<br>
                        Player stability looks strong based on current stats.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Stats Summary Card ────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header' style='font-size:1.3rem'>📋 PLAYER STATS SUMMARY</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub'>All stats entered for this prediction</div>", unsafe_allow_html=True)

            import pandas as pd
            summary_data = {"Stat": [], "Value": []}
            for key, label, *_ in feats:
                summary_data["Stat"].append(label)
                val = input_values[key]
                summary_data["Value"].append(round(val, 2) if isinstance(val, float) else val)
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

                        # ── Feature Importance Chart ──────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header' style='font-size:1.3rem'>📊 FEATURE IMPORTANCE</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub'>Which stats influenced the prediction most</div>", unsafe_allow_html=True)

            try:
                importances = model.feature_importances_
                feat_labels = [f[1] for f in feats]
                sorted_idx  = np.argsort(importances)
                sorted_imp  = importances[sorted_idx]
                sorted_lbl  = [feat_labels[i] for i in sorted_idx]
                bar_colors  = [card_color if v > np.median(sorted_imp) else "#2a4a5a" for v in sorted_imp]

                fig, ax = plt.subplots(figsize=(7, len(feat_labels) * 0.45 + 1))
                fig.patch.set_facecolor('#0d1b2a')
                ax.set_facecolor('#0d1b2a')

                bars = ax.barh(sorted_lbl, sorted_imp, color=bar_colors, height=0.6, edgecolor='none')

                # Value labels on bars
                for bar, val in zip(bars, sorted_imp):
                    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                            f'{val:.3f}', va='center', ha='left',
                            color='#aabbcc', fontsize=8)

                ax.set_xlabel('Importance Score', color='#7799aa', fontsize=9)
                ax.tick_params(colors='#aabbcc', labelsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#1a2a3a')
                ax.spines['bottom'].set_color('#1a2a3a')
                ax.xaxis.label.set_color('#7799aa')
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.info("Feature importance not available for this model.")

    st.markdown("""
    <div style='text-align:center;margin-top:2.5rem;color:#2a3a4a;font-size:0.72rem;letter-spacing:1px;'>
        NOTE: NATIONALITY & CLUB DUMMY FEATURES NOT INCLUDED — CORE STATS ONLY
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "predict":
    page_predict()