# ============================================================
# INTELLIGENT NEWS ANALYZER
# ============================================================

import streamlit as st
import pandas as pd
import re
import plotly.express as px

from database import (
    init_database,
    save_news,
    save_analysis,
    get_all_news
)

from auth import (
    initialize_database,
    create_user,
    login_user
)

from utils import (
    extract_pdf_text,
    preprocess_text,
    classify_news,
    extract_keywords,
    extract_entities,
    analyze_sentiment,
    summarize_text,
    explain_prediction,
    detect_fake_news,
    detect_language,
    translate_to_english,
    find_similar_news,
    calculate_credibility_score,
    get_credibility_level,
    detect_bias,
    detect_claims,
    detect_contradictions
)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

init_database()
initialize_database()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NewsLens AI | News Intelligence",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "page" not in st.session_state:
    st.session_state.page = "Login"


# ============================================================
# PLAIN BACKGROUND (BACKGROUND IMAGE REMOVED)
# ============================================================

def set_background(image_name=None):
    """Apply a consistent, readable dark-blue theme without background images."""
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: #0B1220 !important; color: #F8FAFC !important;
        }
        [data-testid="stHeader"] { background: rgba(11,18,32,.95) !important; }
        .block-container { max-width: 1280px; padding-top: 1.8rem; padding-bottom: 3rem; }
        h1,h2,h3,h4,h5,h6 { color: #F8FAFC !important; font-weight: 750 !important; }
        p,li,label,[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] { color: #E2E8F0; }
        [data-testid="stCaptionContainer"], .stCaption { color: #B8C5D8 !important; }
        [data-testid="stMetric"] { background:#151F32; border:1px solid #293750; padding:16px; border-radius:12px; }
        [data-testid="stMetricLabel"] { color:#B8C5D8 !important; }
        [data-testid="stMetricValue"] { color:#F8FAFC !important; }
        .stTextInput input,.stTextArea textarea,.stNumberInput input,
        [data-baseweb="select"] > div { background:#151F32 !important; color:#F8FAFC !important; border:1px solid #475569 !important; }
        .stTextArea textarea::placeholder,.stTextInput input::placeholder { color:#94A3B8 !important; }
        .stButton > button,.stDownloadButton > button { background:#2563EB !important; color:#FFFFFF !important; border:1px solid #3B82F6 !important; border-radius:9px !important; font-weight:650 !important; min-height:2.5rem; }
        .stButton > button:hover,.stDownloadButton > button:hover { background:#1D4ED8 !important; border-color:#60A5FA !important; }
        [data-testid="stSidebar"] { background:#111827 !important; border-right:1px solid #263349; }
        [data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span { color:#E5E7EB !important; }
        [data-testid="stAlert"] { border-radius:10px; }
        hr { border-color:#334155 !important; }
        a { color:#93C5FD !important; }
        [data-testid="stExpander"] { background:#111C2E; border:1px solid #293750; border-radius:10px; }
        /* Clear typography and modern workspace layout */
        html, body, [class*="css"] { font-family: Inter, "Segoe UI", sans-serif; }
        h1 { letter-spacing: -0.035em; font-size: 2.25rem !important; }
        h2 { letter-spacing: -0.02em; margin-top: 1.1rem !important; }
        [data-testid="stMetric"] { min-height: 112px; border-radius: 14px !important; }
        [data-testid="stAlert"] { border-radius: 12px !important; }
        .stButton > button { min-height: 42px; transition: all .15s ease-in-out; }
        [data-testid="stDataFrame"] { border: 1px solid #293750; border-radius: 12px; overflow: hidden; }
        .block-container { max-width: 1320px; padding-top: 1.6rem; }

        /* NewsLens AI reference-inspired dashboard */
        .block-container { max-width: 1500px !important; padding-top: 1rem !important; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg,#07152B 0%,#0B1730 100%) !important; border-right:1px solid #15355E !important; }
        [data-testid="stSidebar"] .stButton > button { text-align:left !important; justify-content:flex-start !important; min-height:40px !important; background:transparent !important; border-color:transparent !important; color:#C8D8F0 !important; }
        [data-testid="stSidebar"] .stButton > button:hover { background:#123B75 !important; border-color:#245CB5 !important; }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] { background:linear-gradient(100deg,#1263D8,#0D4EAC) !important; border-color:#277DE5 !important; color:white !important; }
        .brand-lockup { display:flex; align-items:center; gap:11px; padding:6px 2px 22px; border-bottom:1px solid #1D3558; margin-bottom:15px; }
        .brand-icon { width:42px;height:42px;display:flex;align-items:center;justify-content:center;border-radius:13px;background:linear-gradient(145deg,#0F6AE8,#20C8F5);box-shadow:0 0 22px #1683FF55;color:white;font-size:27px; }
        .brand-name { font-size:19px;font-weight:850;letter-spacing:-.6px;color:#F8FAFC; }.brand-name span {color:#22B8FF}.brand-sub {font-size:8px;letter-spacing:1.2px;color:#8EAED9;margin-top:2px}
        .sidebar-caption,.nav-caption { color:#6F93C2;font-size:10px;letter-spacing:1.8px;font-weight:800;margin:12px 0 8px; }
        .user-chip { display:flex;align-items:center;gap:10px;background:#102748;border:1px solid #1D4778;border-radius:12px;padding:10px;margin:8px 0 18px;color:#EAF3FF; }
        .user-chip b {display:block;font-size:12px}.user-chip small {display:block;color:#8EAED9;font-size:10px;margin-top:2px}.avatar {width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#2563EB;color:#fff;font-weight:800}
        .sidebar-foot {margin-top:24px;padding:13px;border:1px solid #1B3C65;border-radius:12px;background:#0B203C;color:#AFC6E7;font-size:11px;line-height:1.7}.sidebar-foot small {color:#7898C2}.status-dot {display:inline-block;width:7px;height:7px;border-radius:50%;background:#20D6A2;margin-right:6px;box-shadow:0 0 10px #20D6A2}
        
        .eyebrow,.section-kicker {font-size:10px;letter-spacing:2px;font-weight:850;color:#29B9FF}
        
        .dashboard-project-title {display:flex;align-items:center;gap:12px;margin:2px 0 12px;padding:0 0 13px;border-bottom:1px solid #19375C;color:#F8FAFC;font-size:30px;line-height:1.15;font-weight:850;letter-spacing:-.8px}.dashboard-project-title .project-mark {display:inline-flex;align-items:center;justify-content:center;width:42px;height:42px;border-radius:12px;background:linear-gradient(145deg,#0F6AE8,#20C8F5);box-shadow:0 0 20px #1683FF35;font-size:25px}.dashboard-project-title .project-name-accent {color:#29B9FF}.dashboard-project-subtitle {font-size:11px;font-weight:450;letter-spacing:0;color:#8EA6C8;margin-top:5px}
        .welcome-line {color:#EAF3FF;font-size:13px;margin:14px 0 12px}.welcome-line span {color:#8EA6C8;margin-left:8px;font-size:11px}.metric-grid {margin:4px 0 18px}.metric-card {height:108px;background:linear-gradient(145deg,#0F213C,#0B172C);border:1px solid #1D416D;border-radius:14px;padding:13px;position:relative;box-shadow:0 5px 18px #02081722}.metric-icon {height:37px;width:37px;border-radius:12px;border:1px solid;display:flex;align-items:center;justify-content:center;font-size:20px;float:left;margin-right:10px}.metric-value {font-size:25px;font-weight:850;line-height:1.2;color:#F7FBFF;padding-top:2px}.metric-label {clear:both;padding-top:8px;color:#AFC4E4;font-size:10px;line-height:1.3}
        @media(max-width:700px){.dashboard-project-title{font-size:23px}.dashboard-project-title .project-mark{width:36px;height:36px;font-size:21px}}
        .section-heading {display:flex;justify-content:space-between;align-items:center;margin:6px 0 12px;padding:14px 16px 0;border-top:1px solid #19375C}.section-heading h3 {font-size:19px!important;margin:3px 0 0!important}.action-heading {margin-top:20px}.empty-state {min-height:180px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;background:#0D1A2E;border:1px dashed #25476F;border-radius:14px;padding:22px;color:#D8E8FC}.empty-state p {font-size:11px;color:#91A9CA!important;max-width:260px}.empty-icon {font-size:30px;color:#29B9FF;margin-bottom:10px}.action-card {height:142px;border:1px solid color-mix(in srgb,var(--action-color) 70%,#1A3456);background:linear-gradient(140deg,color-mix(in srgb,var(--action-color) 28%,#09152A),#0A172B);border-radius:15px;padding:14px;margin-bottom:8px}.action-icon {display:flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:12px;background:var(--action-color);color:white;font-size:23px;box-shadow:0 5px 20px color-mix(in srgb,var(--action-color) 40%,transparent)}.action-card h4 {font-size:14px!important;margin:9px 0 4px!important}.action-card p {font-size:10px;line-height:1.45;color:#B5C8E4!important;margin:0}.connection-card {padding:16px;margin:10px 0;border:1px solid #25476F;border-radius:13px;background:linear-gradient(120deg,#0E2039,#0B1729)}.connection-card p {color:#B8CBE7!important;font-size:12px;line-height:1.6}.similarity-pill {float:right;border-radius:999px;padding:4px 8px;background:#0D4C7C;color:#8EE7FF;font-size:11px;font-weight:800}
        [data-testid="stDataFrame"] {background:#0C192C!important}.stPlotlyChart {border:1px solid #19375C;border-radius:14px;background:#0A172A;padding:6px}
        @media(max-width:900px){.metric-card{margin-bottom:8px}}

        </style>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    set_background("news_background.jpg")

    st.markdown(
        """
        <h1 style="
            text-align:center;
            font-size:45px;
        ">
            📰 Intelligent News Analyzer
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style="
            text-align:center;
            font-size:18px;
        ">
            AI-Powered News Categorization & Analysis
        </p>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    left, center, right = st.columns(
        [1, 2, 1]
    )

    with center:

        st.subheader("🔐 Login")

        phone = st.text_input(
            "📱 Phone Number",
            placeholder="+919876543210"
        )

        password = st.text_input(
            "🔑 Password",
            type="password"
        )

        if st.button(
            "🚀 Login",
            width="stretch"
        ):

            if not phone or not password:

                st.warning(
                    "Please enter phone number and password."
                )

            else:

                user = login_user(
                    phone,
                    password
                )

                if user:

                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.session_state.page = "Dashboard"

                    st.success(
                        "Login successful!"
                    )

                    st.rerun()

                else:

                    st.error(
                        "Invalid phone number or password."
                    )

        st.write("")

        st.write(
            "Don't have an account?"
        )

        if st.button(
            "📝 Create New Account",
            width="stretch"
        ):

            st.session_state.page = "Sign Up"

            st.rerun()


# ============================================================
# SIGN UP PAGE
# ============================================================

def signup_page():

    set_background(
        "news_background.jpg"
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            padding-top:30px;
        ">

            <h1 style="
                font-size:40px;
                font-weight:bold;
            ">
                📝 Create Your Account
            </h1>

            <p style="
                font-size:18px;
            ">
                Join the Intelligent News Analyzer
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    left, center, right = st.columns(
        [1, 2, 1]
    )

    with center:

        st.subheader(
            "👤 Account Details"
        )

        name = st.text_input(
            "👤 Full Name",
            placeholder="Enter your name",
            key="signup_name"
        )

        phone = st.text_input(
            "📱 Phone Number",
            placeholder="+919876543210",
            key="signup_phone"
        )

        password = st.text_input(
            "🔑 Password",
            type="password",
            placeholder="Minimum 6 characters",
            key="signup_password"
        )

        confirm_password = st.text_input(
            "🔑 Confirm Password",
            type="password",
            key="signup_confirm_password"
        )

        signup_button = st.button(
            "✨ Create Account",
            width="stretch"
        )

        if signup_button:

            if (
                not name
                or not phone
                or not password
                or not confirm_password
            ):

                st.warning(
                    "Please fill in all fields."
                )

            elif password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            elif len(password) < 6:

                st.error(
                    "Password must contain at least 6 characters."
                )

            else:

                success, message = create_user(
                    name,
                    phone,
                    password
                )

                if success:

                    st.success(
                        message
                    )

                    st.info(
                        "Your account has been created. "
                        "Please login."
                    )

                else:

                    st.error(
                        message
                    )

        st.write("")

        st.write(
            "Already have an account?"
        )

        if st.button(
            "🔐 Back to Login",
            width="stretch"
        ):

            st.session_state.page = "Login"

            st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():
    """Consistent app navigation; pages are wired to working functionality."""
    st.sidebar.markdown("""<div class="sidebar-caption">WORKSPACE</div>""", unsafe_allow_html=True)
    if st.session_state.user:
        st.sidebar.markdown(
            f"<div class='user-chip'><div class='avatar'>{str(st.session_state.user.get('name','U'))[:1].upper()}</div><div><b>{st.session_state.user.get('name','User')}</b><small>News analyst</small></div></div>",
            unsafe_allow_html=True,
        )
    pages = [
        ("Dashboard", "⌂", "Overview"),
        ("Analyzer", "▤", "Analyze News"),
        ("History", "◷", "History & Reports"),
        ("Connections", "⌘", "News Connections"),
        ("Claim Explorer", "⌕", "Claim Explorer"),
        ("Profile", "⚙", "Settings & Profile"),
    ]
    st.sidebar.markdown('<div class="nav-caption">NAVIGATION</div>', unsafe_allow_html=True)
    for page_key, symbol, label in pages:
        active = st.session_state.page == page_key
        if st.sidebar.button(f"{symbol}   {label}", key=f"nav_{page_key}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.page = page_key
            st.rerun()
    st.sidebar.markdown("<div class='sidebar-foot'><span class='status-dot'></span> NLP engine workspace<br><small>Predictions are indicators, not fact verification.</small></div>", unsafe_allow_html=True)
    if st.sidebar.button("↪   Sign out", key="nav_logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.page = "Login"
        st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():
    """NewsLens AI overview dashboard inspired by the user's reference image."""
    set_background()
    sidebar()
    # Isolated HTML brand header. This avoids the SVG/text rendering issue
    # that made a horizontal line appear to cross the project title.
    st.markdown(
        """
        <style>
        .nl-brand-header {
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
            width: 100% !important;
            min-height: 58px !important;
            margin: 0 !important;
            padding: 2px 0 12px !important;
            border: 0 !important;
            border-bottom: 1px solid #19375C !important;
            background: transparent !important;
            position: relative !important;
            isolation: isolate !important;
            overflow: visible !important;
            opacity: 1 !important;
            filter: none !important;
            mix-blend-mode: normal !important;
            text-decoration: none !important;
        }
        .nl-brand-header::before,
        .nl-brand-header::after,
        .nl-brand-copy::before,
        .nl-brand-copy::after {
            content: none !important;
            display: none !important;
        }
        .nl-brand-icon {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            flex: 0 0 42px !important;
            width: 42px !important;
            height: 42px !important;
            border-radius: 12px !important;
            background: linear-gradient(145deg, #0F6AE8, #20C8F5) !important;
            color: #FFFFFF !important;
            font-size: 24px !important;
            line-height: 1 !important;
            opacity: 1 !important;
            position: relative !important;
            z-index: 1 !important;
        }
        .nl-brand-copy {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            gap: 4px !important;
            min-width: 0 !important;
            position: relative !important;
            z-index: 2 !important;
            opacity: 1 !important;
            filter: none !important;
            mix-blend-mode: normal !important;
            text-decoration: none !important;
            background: transparent !important;
        }
        .nl-brand-title {
            display: block !important;
            margin: 0 !important;
            padding: 0 !important;
            color: #F8FAFC !important;
            font-family: Inter, 'Segoe UI', Arial, sans-serif !important;
            font-size: 28px !important;
            font-weight: 850 !important;
            letter-spacing: -0.8px !important;
            line-height: 1.15 !important;
            white-space: normal !important;
            opacity: 1 !important;
            filter: none !important;
            mix-blend-mode: normal !important;
            text-decoration: none !important;
            text-shadow: none !important;
            background: transparent !important;
            position: relative !important;
            z-index: 3 !important;
        }
        .nl-brand-title span { color: #29B9FF !important; }
        .nl-brand-subtitle {
            display: block !important;
            margin: 0 !important;
            padding: 0 !important;
            color: #B8CBE7 !important;
            font-family: Inter, 'Segoe UI', Arial, sans-serif !important;
            font-size: 11px !important;
            font-weight: 400 !important;
            line-height: 1.35 !important;
            letter-spacing: 0 !important;
            opacity: 1 !important;
            filter: none !important;
            text-decoration: none !important;
            background: transparent !important;
        }
        @media (max-width: 700px) {
            .nl-brand-title { font-size: 22px !important; }
            .nl-brand-icon { flex-basis: 36px !important; width: 36px !important; height: 36px !important; }
        }
        </style>
        <div class="nl-brand-header">
          <div class="nl-brand-icon" aria-hidden="true">▤</div>
          <div class="nl-brand-copy">
            <div class="nl-brand-title">Intelligent <span>News Analyzer</span></div>
            <div class="nl-brand-subtitle">Multilingual news intelligence and explainable NLP analysis</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_name = (st.session_state.user or {}).get("name", "Analyst")
    st.markdown(f"<div class='welcome-line'>Welcome back, <b>{user_name}</b> <span>Here is your news workspace at a glance.</span></div>", unsafe_allow_html=True)

    # Load the existing history store; all metrics are based on actual rows.
    try:
        history = get_all_news()
        if not isinstance(history, pd.DataFrame):
            history = pd.DataFrame(history or [])
    except Exception as exc:
        history = pd.DataFrame()
        st.warning(f"History metrics are temporarily unavailable: {exc}")

    if not history.empty and st.session_state.user and "user_id" in history.columns:
        history = history[history["user_id"].astype(str) == str(st.session_state.user.get("id"))].copy()
    total_articles = len(history)
    languages_count = int(history["language"].dropna().nunique()) if "language" in history.columns else 0
    # Build category distribution from saved analysis categories, with the
    # article title as a backward-compatible fallback for older database rows.
    supported_categories = [
        "World", "Sports", "Business", "Science/Technology",
        "Environment", "Education", "Agriculture", "Health",
        "Politics/Government",
    ]

    def _normalise_saved_category(value):
        if value is None or pd.isna(value):
            return None
        raw = re.sub(r"\s+", " ", str(value)).strip()
        aliases = {
            "science/technology": "Science/Technology",
            "science & technology": "Science/Technology",
            "science and technology": "Science/Technology",
            "politics": "Politics/Government",
            "government": "Politics/Government",
            "politics/government": "Politics/Government",
        }
        lowered = raw.lower()
        if lowered.endswith(" news"):
            lowered = lowered[:-5].strip()
        if lowered in aliases:
            return aliases[lowered]
        for category in supported_categories:
            if lowered == category.lower():
                return category
        return None

    category_values = []
    if not history.empty:
        for _, history_row in history.iterrows():
            # Titles are written as '<predicted category> News' when an
            # analysis is saved, so prefer that label for legacy rows where
            # the joined analysis category may be missing or stale.
            saved_category = _normalise_saved_category(
                history_row.get("title") if "title" in history.columns else None
            )
            if saved_category is None:
                saved_category = _normalise_saved_category(
                    history_row.get("category") if "category" in history.columns else None
                )
            if saved_category:
                category_values.append(saved_category)

    category_count = len(set(category_values))

    # Count saved articles whose fake-news model explicitly flagged them
    # for human verification. The count updates automatically when another
    # flagged article is saved to history; no placeholder text is displayed.
    review_count = 0
    if not history.empty and "fake_prediction" in history.columns:
        review_labels = {
            "potentially fake", "needs review", "review required",
            "uncertain", "inconclusive", "unknown"
        }
        review_count = int(
            history["fake_prediction"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin(review_labels)
            .sum()
        )

    # Claims are computed per article only in analysis; never fabricate a dashboard count.
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    metric_data = [
        ("▤", str(total_articles), "Articles analyzed", "#28B8F2"),
        ("◎", str(max(languages_count, 0)), "Languages in history", "#22D3EE"),
        ("◆", str(category_count), "News categories", "#FBBF24"),
        ("✧", "5", "AI analysis modules", "#A78BFA"),
        ("!", str(review_count), "Articles needing review", "#FB7185"),
    ]
    cols = st.columns(5)
    for col, (icon, value, label, color) in zip(cols, metric_data):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-icon' style='color:{color};border-color:{color}44;background:{color}15'>{icon}</div><div class='metric-value'>{value}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("<div class='section-heading'><div><span class='section-kicker'>YOUR WORKSPACE</span><h3>Recent analyses</h3></div></div>", unsafe_allow_html=True)
        if not history.empty:
            recent = history.copy()
            if "created_at" in recent.columns:
                recent = recent.sort_values("created_at", ascending=False, na_position="last")
            show_cols = []
            for col_name in ["title", "category", "language", "created_at"]:
                if col_name in recent.columns:
                    show_cols.append(col_name)
            if "title" not in show_cols and "article_text" in recent.columns:
                recent["title"] = recent["article_text"].fillna("").str.slice(0, 70).replace("", "Untitled article")
                show_cols.insert(0, "title")
            if show_cols:
                view = recent[show_cols].head(6).copy()
                if "category" in view.columns:
                    for row_index in view.index:
                        fallback_category = _normalise_saved_category(
                            view.at[row_index, "title"] if "title" in view.columns else None
                        )
                        if fallback_category:
                            view.at[row_index, "category"] = fallback_category
                        elif _normalise_saved_category(view.at[row_index, "category"]) is None:
                            view.at[row_index, "category"] = "Uncategorized"
                # Convert stored ISO language codes to readable names for display.
                # Keep the underlying history/database values unchanged.
                language_names = {
                    "ar": "Arabic", "bn": "Bengali", "zh-cn": "Chinese (Simplified)",
                    "zh-tw": "Chinese (Traditional)", "en": "English", "fr": "French",
                    "de": "German", "gu": "Gujarati", "hi": "Hindi", "id": "Indonesian",
                    "it": "Italian", "ja": "Japanese", "kn": "Kannada", "ko": "Korean",
                    "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali", "pa": "Punjabi",
                    "pt": "Portuguese", "ru": "Russian", "es": "Spanish", "ta": "Tamil",
                    "te": "Telugu", "th": "Thai", "ur": "Urdu", "vi": "Vietnamese",
                }
                if "language" in view.columns:
                    view["language"] = view["language"].apply(
                        lambda value: language_names.get(str(value).strip().lower(), str(value).strip().title())
                        if pd.notna(value) and str(value).strip() else "Unknown"
                    )
                view.columns = ["Article" if c == "title" else "Category" if c == "category" else "Language" if c == "language" else "Analyzed" if c == "created_at" else c for c in view.columns]
                st.dataframe(view, use_container_width=True, hide_index=True, height=265)
            else:
                st.info("Your previous articles are saved. Open History & Reports to inspect them.")
        else:
            st.markdown("<div class='empty-state'><div class='empty-icon'>▤</div><b>Your story timeline starts here</b><p>Analyze your first article to build a personal research history.</p></div>", unsafe_allow_html=True)
        if st.button("View all reports  →", key="dashboard_history", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()
    with right:
        st.markdown("<div class='section-heading'><div><span class='section-kicker'>INSIGHT SNAPSHOT</span><h3>Category distribution</h3></div></div>", unsafe_allow_html=True)
        if category_values:
            cat_counts = pd.Series(category_values).value_counts().rename_axis("Category").reset_index(name="Articles")
            category_colors = {
                "World": "#2585FF", "Sports": "#FB7185", "Business": "#8B5CF6",
                "Science/Technology": "#FBBF24", "Environment": "#19C995",
                "Education": "#38BDF8", "Agriculture": "#84CC16",
                "Health": "#F97316", "Politics/Government": "#E879F9",
            }
            fig = px.pie(
                cat_counts, names="Category", values="Articles", hole=0.68,
                color="Category", color_discrete_map=category_colors,
            )
            fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="#0B1220", width=3)))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#E6F0FF", margin=dict(l=8,r=8,t=8,b=8), height=260,
                              legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0),
                              showlegend=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("<div class='empty-state'><div class='empty-icon'>◔</div><b>Insights appear as you analyze</b><p>Your category chart will use saved category labels when available.</p></div>", unsafe_allow_html=True)
        st.caption(
            f"Distribution of {len(category_values)} saved article categorization(s). "
            "The chart updates when new analyses are saved and the dashboard is refreshed. "
            "Older records without a saved category are inferred from their title when possible."
        )

    st.markdown("<div class='section-heading action-heading'><div><span class='section-kicker'>QUICK ACTIONS</span><h3>What would you like to explore?</h3></div></div>", unsafe_allow_html=True)
    action_cols = st.columns(4)
    actions = [
        ("▤", "Analyze news", "Paste text or upload a PDF.", "Analyzer", "#0B4FA8"),
        ("⌕", "Claim Explorer", "Inspect claims and possible conflicts.", "Claim Explorer", "#43228A"),
        ("⌘", "News connections", "Find related articles across history.", "Connections", "#064E45"),
        ("◷", "History & reports", "Return to previous analyses.", "History", "#5A3B0B"),
    ]
    for col, (icon, title, desc, page, color) in zip(action_cols, actions):
        with col:
            st.markdown(f"<div class='action-card' style='--action-color:{color}'><div class='action-icon'>{icon}</div><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)
            if st.button(f"Open {title} →", key=f"quick_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()



def history_page():
    set_background()
    sidebar()
    st.title("◷ History & Reports")
    st.caption("Browse saved articles and reopen text for further analysis.")
    try:
        data = get_all_news()
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data or [])
    except Exception as exc:
        st.error(f"Could not load saved history: {exc}")
        return
    if not data.empty and st.session_state.user and "user_id" in data.columns:
        data = data[data["user_id"].astype(str) == str(st.session_state.user.get("id"))]
    if data.empty:
        st.info("No saved articles yet. Analyze an article first to create your history.")
        if st.button("Analyze your first article", type="primary"):
            st.session_state.page = "Analyzer"; st.rerun()
        return
    search = st.text_input("Search saved articles", placeholder="Search title or article text...")
    filtered = data.copy()
    if search.strip():
        mask = pd.Series(False, index=filtered.index)
        for col in ["title", "article_text", "language"]:
            if col in filtered.columns:
                mask = mask | filtered[col].fillna("").astype(str).str.contains(search, case=False, regex=False)
        filtered = filtered[mask]
    display_cols = [c for c in ["title", "language", "created_at"] if c in filtered.columns]
    if display_cols:
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    options = filtered.index.tolist()
    if options and "article_text" in filtered.columns:
        chosen = st.selectbox("Select an article to inspect", options, format_func=lambda idx: str(filtered.loc[idx].get("title", "Saved article")))
        with st.expander("Read selected article"):
            st.write(str(filtered.loc[chosen].get("article_text", "")))
        if st.button("Analyze selected article again", type="primary"):
            st.session_state["prefill_article"] = str(filtered.loc[chosen].get("article_text", ""))
            st.session_state.page = "Analyzer"; st.rerun()


def connections_page():
    set_background()
    sidebar()
    st.title("⌘ News Connections")
    st.caption("Explore semantic connections between this article and previously saved news.")
    try:
        data = get_all_news()
        if not isinstance(data, pd.DataFrame): data = pd.DataFrame(data or [])
    except Exception as exc:
        st.error(f"Could not load news history: {exc}"); return
    if not data.empty and st.session_state.user and "user_id" in data.columns:
        data = data[data["user_id"].astype(str) == str(st.session_state.user.get("id"))]
    if data.empty or "article_text" not in data.columns:
        st.info("Analyze and save at least two articles to explore related stories."); return
    rows = data[data["article_text"].fillna("").astype(str).str.strip().ne("")]
    if len(rows) < 2:
        st.info("Analyze at least two articles to create a useful connection map."); return
    choices = rows.index.tolist()
    selected = st.selectbox("Choose a source article", choices, format_func=lambda idx: str(rows.loc[idx].get("title", "Saved article")))
    source_text = str(rows.loc[selected]["article_text"])
    if st.button("Find related stories", type="primary"):
        try:
            matches = find_similar_news(source_text, rows.rename(columns={"article_text":"text"}), top_n=6)
            if isinstance(matches, pd.DataFrame) and not matches.empty:
                st.subheader("Related coverage")
                for _, item in matches.iterrows():
                    similarity = float(item.get("similarity", 0) or 0) * 100
                    st.markdown(f"<div class='connection-card'><b>{item.get('title','Related article')}</b><span class='similarity-pill'>{similarity:.1f}% similar</span><p>{str(item.get('text',''))[:450]}</p></div>", unsafe_allow_html=True)
            else: st.info("No meaningfully similar articles were found.")
        except Exception as exc:
            st.error(f"Related-story search failed: {exc}")


def claim_explorer_page():
    """Render claim and contradiction analysis as readable UI, not raw dictionaries."""
    set_background()
    sidebar()

    st.title("⌕ Claim Explorer")
    st.caption(
        "Review potentially verifiable statements and possible internal conflicts. "
        "These are automated review signals, not fact-check verdicts."
    )

    with st.container(border=True):
        st.subheader("Article to inspect")
        article = st.text_area(
            "Paste an article",
            key="claim_explorer_input",
            height=180,
            placeholder="Paste a news report to identify claims and possible conflicts...",
            label_visibility="visible",
        )
        explore_clicked = st.button(
            "🔎 Explore claims",
            type="primary",
            width="stretch",
        )

    if not explore_clicked:
        st.info("Paste an article above and select **Explore claims** to see a readable summary.")
        return

    if not article or not article.strip():
        st.warning("Enter an article first.")
        return

    try:
        with st.spinner("Checking statements and comparing possible conflicts..."):
            clean = preprocess_text(article)
            claims_result = detect_claims(clean)
            contradictions_result = detect_contradictions(clean)

        # Normalize the expected result structures without exposing raw dictionaries.
        if not isinstance(claims_result, dict):
            claims_result = {"claims": claims_result if isinstance(claims_result, list) else []}
        if not isinstance(contradictions_result, dict):
            contradictions_result = {
                "contradictions": contradictions_result if isinstance(contradictions_result, list) else []
            }

        claim_items = claims_result.get("claims", []) or []
        contradiction_items = contradictions_result.get("contradictions", []) or []
        claim_items = [item for item in claim_items if isinstance(item, dict)]
        contradiction_items = [item for item in contradiction_items if isinstance(item, dict)]

        st.markdown("## Analysis overview")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Sentences checked", claims_result.get("total_sentences", "—"))
        metric_cols[1].metric("Claim candidates", claims_result.get("claim_count", len(claim_items)))
        claim_score = claims_result.get("claim_score")
        metric_cols[2].metric("Claim presence score", f"{claim_score}%" if isinstance(claim_score, (int, float)) else "—")
        metric_cols[3].metric("Potential conflicts", contradictions_result.get("contradiction_count", len(contradiction_items)))

        claim_level = str(claims_result.get("level", "Not available"))
        contradiction_level = str(contradictions_result.get("level", "Not available"))
        summary_cols = st.columns(2)
        with summary_cols[0]:
            st.markdown("#### Claim assessment")
            st.info(claim_level)
        with summary_cols[1]:
            st.markdown("#### Contradiction assessment")
            if contradiction_items:
                st.warning(contradiction_level)
            else:
                st.success(contradiction_level)

        st.divider()
        st.markdown("## Claim candidates")
        st.caption("Statements selected by the claim-detection rules. Confidence is a model/rule score, not the probability that a statement is true.")

        if claim_items:
            for index, item in enumerate(claim_items, start=1):
                with st.container(border=True):
                    heading_cols = st.columns([5, 1])
                    with heading_cols[0]:
                        st.markdown(f"**Claim {index}**")
                    confidence = item.get("confidence")
                    with heading_cols[1]:
                        st.metric("Signal score", f"{confidence}%" if isinstance(confidence, (int, float)) else "—")
                    claim_text = str(item.get("claim", "Statement text unavailable"))
                    st.write(claim_text)
                    if isinstance(confidence, (int, float)):
                        st.progress(max(0.0, min(1.0, float(confidence) / 100.0)))

                    reasons = item.get("reasons", []) or []
                    cue_words = item.get("cue_words", []) or []
                    numbers = item.get("numbers", []) or []
                    entities = item.get("entities", []) or []
                    detail_cols = st.columns(2)
                    with detail_cols[0]:
                        st.markdown("**Why it was selected**")
                        if reasons:
                            for reason in reasons:
                                st.markdown(f"- {reason}")
                        else:
                            st.caption("No additional selection reasons were provided.")
                        if cue_words:
                            st.markdown("**Signal words:** " + ", ".join(map(str, cue_words)))
                    with detail_cols[1]:
                        if numbers:
                            st.markdown("**Numbers / dates:** " + ", ".join(map(str, numbers)))
                        if entities:
                            st.markdown("**Named entities:** " + ", ".join(map(str, entities)))
                        if not numbers and not entities:
                            st.caption("No numbers, dates, or named entities were extracted for this statement.")
        else:
            st.success("No strong claim candidates were detected.")
            st.caption("This does not prove the article contains no factual claims; some statements may be missed by automated rules.")

        st.divider()
        st.markdown("## Potential contradictions")
        st.caption("The system compares statement pairs for possible disagreement. Review the original context before drawing conclusions.")

        if contradiction_items:
            for index, item in enumerate(contradiction_items, start=1):
                with st.container(border=True):
                    st.markdown(f"**Potential conflict {index}**")
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**Statement A**")
                        st.write(str(item.get("statement_1", "Statement unavailable")))
                    with right:
                        st.markdown("**Statement B**")
                        st.write(str(item.get("statement_2", "Statement unavailable")))
                    confidence = item.get("confidence")
                    if isinstance(confidence, (int, float)):
                        st.metric("Contradiction signal", f"{confidence}%")
                        st.progress(max(0.0, min(1.0, float(confidence) / 100.0)))
                    score_cols = st.columns(2)
                    entailment = item.get("entailment_score")
                    neutral = item.get("neutral_score")
                    score_cols[0].metric("Entailment signal", f"{entailment}%" if isinstance(entailment, (int, float)) else "—")
                    score_cols[1].metric("Neutral signal", f"{neutral}%" if isinstance(neutral, (int, float)) else "—")
        else:
            st.success("No strong internal contradictions were detected by the current checks.")
            st.caption("A result with no detected conflicts is not proof that every statement is consistent or factually correct.")

        st.divider()
        st.warning(
            "Important: automated checks can miss conflicts or flag unrelated statements. "
            "Verify important claims against reliable, independent sources."
        )

    except Exception as exc:
        st.error(f"Claim exploration failed: {exc}")


# ============================================================
# NEWS ANALYZER
# ============================================================

def analyzer_page():

    set_background(
        "news_background.jpg"
    )

    sidebar()

    st.title(
        "📰 Intelligent News Analyzer"
    )

    st.caption(
        "Analyze, classify and explain a news article using NLP and AI."
    )

    # --------------------------------------------------------
    # INPUT METHOD
    # --------------------------------------------------------

    input_method = st.radio(
        "Choose input method",
        [
            "Paste Text",
            "Upload PDF"
        ],
        horizontal=True
    )

    text = st.session_state.pop("prefill_article", "")

    # --------------------------------------------------------
    # PASTE TEXT
    # --------------------------------------------------------

    if input_method == "Paste Text":

        text = st.text_area(
            "Paste your news article here:",
            value=text,
            height=250,
            placeholder=(
                "Example: Apple has announced a new "
                "artificial intelligence processor..."
            )
        )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    else:

        uploaded_file = st.file_uploader(
            "Upload a news article PDF",
            type=["pdf"]
        )

        if uploaded_file:

            with st.spinner(
                "Extracting text from PDF..."
            ):

                text = extract_pdf_text(
                    uploaded_file
                )

            if text and text.strip():

                st.success(
                    "PDF text extracted successfully."
                )

            else:

                st.error(
                    "No readable text was found in the PDF."
                )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze News",
        type="primary",
        width="stretch"
    ):

        if not text or not text.strip():

            st.warning(
                "Please enter or upload a news article."
            )

            return

        # ====================================================
        # LANGUAGE DETECTION
        # ====================================================

        with st.spinner(
            "Detecting language..."
        ):

            detected_language = detect_language(
                text
            )

        st.info(
            f"🌐 Detected Language: "
            f"**{detected_language}**"
        )

        # ====================================================
        # MULTILINGUAL TRANSLATION
        # ====================================================

        analysis_text = text

        if detected_language != "English":

            with st.spinner(
                "Translating article to English..."
            ):

                analysis_text = translate_to_english(
                    text
                )

            if analysis_text:

                st.subheader(
                    "🌐 English Translation"
                )

                st.info(
                    analysis_text
                )

            else:

                st.error(
                    "❌ Translation failed. "
                    "Please check your internet connection "
                    "and try again."
                )

                return

        # ====================================================
        # PREPROCESSING
        # ====================================================

        with st.spinner(
            "Preprocessing text..."
        ):

            processed_text = preprocess_text(
                analysis_text
            )

        if not processed_text:

            st.warning(
                "The article does not contain enough readable text."
            )

            return

        # Display a short preview of the exact text used for every model.
        with st.expander("View text sent to the analysis models"):
            st.write(processed_text[:1200])
            st.caption(f"Input length: {len(processed_text)} characters")

        # ====================================================
        # NEWS CLASSIFICATION
        # ====================================================

        st.divider()

        st.header(
            "🏷️ News Classification"
        )

        with st.spinner(
            "Classifying news..."
        ):

            categories = classify_news(
                processed_text
            )

        if not categories:

            st.error(
                "Unable to classify the article."
            )

            return

        top_category = categories[0]

        # ----------------------------------------------------
        # TOP PREDICTION
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Category",
                top_category["category"]
            )

        with col2:

            st.metric(
                "Confidence",
                f"{top_category['score'] * 100:.2f}%"
            )

        # ----------------------------------------------------
        # CONFIDENCE CHART
        # ----------------------------------------------------

        if top_category["category"] == "Environment":

            st.success(
                "🌱 Environmental content detected. "
                "The article contains strong domain-specific "
                "environmental signals."
            )

        st.caption(
            "Predictions combine the four broad AG News labels with "
            "domain-specific signals for Environment, Education, "
            "Agriculture, Health, and Politics/Government. Domain labels "
            "are keyword-assisted and should be reviewed for ambiguous articles."
        )

        st.subheader(
            "📊 Category Confidence"
        )

        confidence_df = pd.DataFrame(
            categories
        )

        confidence_df["score"] = (
            confidence_df["score"] * 100
        )

        fig = px.bar(
            confidence_df,
            x="category",
            y="score",
            title="News Category Confidence",
            labels={
                "category": "Category",
                "score": "Confidence (%)"
            }
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        st.divider()

        st.header(
            "📝 News Summary"
        )

        with st.spinner(
            "Generating summary..."
        ):

            summary = summarize_text(
                processed_text
            )

        st.info(
            summary
        )

        # ====================================================
        # KEYWORDS
        # ====================================================

        st.header(
            "🔑 Important Keywords"
        )

        with st.spinner(
            "Extracting keywords..."
        ):

            keywords = extract_keywords(
                processed_text
            )

        if keywords:

            st.success(
                " • ".join(keywords)
            )

        else:

            st.info(
                "No keywords detected."
            )

        # ====================================================
        # NAMED ENTITY RECOGNITION
        # ====================================================

        st.header(
            "👤 Named Entities"
        )

        with st.spinner(
            "Detecting named entities..."
        ):

            entities = extract_entities(
                processed_text
            )

        if entities:

            entity_df = pd.DataFrame(
                entities
            )

            st.dataframe(
                entity_df,
                width="stretch"
            )

        else:

            st.info(
                "No named entities detected."
            )

        # ====================================================
        # SENTIMENT ANALYSIS
        # ====================================================

        st.header(
            "😊 Sentiment Analysis"
        )

        with st.spinner(
            "Analyzing sentiment..."
        ):

            sentiment = analyze_sentiment(
                processed_text
            )

        sentiment_label = "Unknown"
        sentiment_score = 0.0

        if sentiment:

            sentiment_label = sentiment.get(
                "sentiment",
                "Unknown"
            )

            sentiment_score = sentiment.get(
                "score",
                0.0
            )

            st.metric(
                sentiment_label,
                f"{sentiment_score * 100:.2f}%"
            )

        # ====================================================
        # BIAS DETECTION
        # ====================================================

        st.divider()

        st.header(
            "⚖️ Bias Detection"
        )

        st.write(
            "Detect potentially biased linguistic patterns such "
            "as loaded words, subjective wording, intensifiers, "
            "fear-based language, and excessive emphasis."
        )

        with st.spinner(
            "Analyzing possible bias..."
        ):

            bias_result = detect_bias(
                processed_text
            )

        bias_score = float(
            bias_result.get(
                "score",
                0.0
            )
        )

        bias_level = bias_result.get(
            "level",
            "Low"
        )

        bias_terms = bias_result.get(
            "terms",
            []
        )

        bias_indicators = bias_result.get(
            "indicators",
            []
        )

        bias_sentences = bias_result.get(
            "sentences",
            []
        )

        bias_col1, bias_col2, bias_col3 = st.columns(3)

        with bias_col1:

            st.metric(
                "Bias Score",
                f"{bias_score:.2f}/100"
            )

        with bias_col2:

            st.metric(
                "Bias Level",
                bias_level
            )

        with bias_col3:

            if bias_score < 20:
                bias_status = "🟢 Low"
            elif bias_score < 45:
                bias_status = "🟡 Moderate"
            elif bias_score < 70:
                bias_status = "🟠 High"
            else:
                bias_status = "🔴 Very High"

            st.metric(
                "Bias Indicator",
                bias_status
            )

        st.progress(
            int(
                max(
                    0,
                    min(
                        100,
                        round(bias_score)
                    )
                )
            )
        )

        if bias_indicators:

            st.subheader(
                "🚩 Detected Bias Indicators"
            )

            for indicator in bias_indicators:

                st.write(
                    f"• {indicator}"
                )

        else:

            st.success(
                "🟢 No strong linguistic bias indicators were detected."
            )

        if bias_terms:

            st.subheader(
                "🔍 Potentially Biased Words / Phrases"
            )

            st.info(
                " • ".join(bias_terms)
            )

        if bias_sentences:

            st.subheader(
                "📌 Sentences Requiring Attention"
            )

            for item in bias_sentences:

                st.warning(
                    f"**Detected terms:** "
                    f"{', '.join(item['terms'])}\n\n"
                    f"**Sentence:** {item['sentence']}"
                )

        st.caption(
            "⚠️ This module detects linguistic indicators of "
            "potential bias. It does not prove the author's intent "
            "or determine objective truth."
        )

        # ====================================================
        # CLAIM DETECTION
        # ====================================================

        st.divider()

        st.header(
            "🔍 Claim Detection"
        )

        st.write(
            "Identify sentences that are likely to contain "
            "factual or verifiable claims. The module highlights "
            "the linguistic evidence used to identify each claim."
        )

        with st.spinner(
            "Detecting factual claim candidates..."
        ):

            claim_result = detect_claims(
                processed_text
            )

        claim_score = float(
            claim_result.get(
                "claim_score",
                0.0
            )
        )

        claim_count = int(
            claim_result.get(
                "claim_count",
                0
            )
        )

        total_sentences = int(
            claim_result.get(
                "total_sentences",
                0
            )
        )

        claim_level = claim_result.get(
            "level",
            "No Claims Detected"
        )

        detected_claims = claim_result.get(
            "claims",
            []
        )

        claim_col1, claim_col2, claim_col3 = st.columns(3)

        with claim_col1:

            st.metric(
                "Claims Detected",
                claim_count
            )

        with claim_col2:

            st.metric(
                "Claim Score",
                f"{claim_score:.2f}/100"
            )

        with claim_col3:

            st.metric(
                "Claim Level",
                claim_level
            )

        if total_sentences > 0:

            st.caption(
                f"Analyzed {total_sentences} sentence(s) "
                f"and identified {claim_count} likely claim(s)."
            )

        if detected_claims:

            st.subheader(
                "📌 Detected Claim Candidates"
            )

            for index, claim in enumerate(
                detected_claims,
                start=1
            ):

                confidence = claim.get(
                    "confidence",
                    0.0
                )

                reasons = claim.get(
                    "reasons",
                    []
                )

                with st.expander(
                    f"Claim {index} — "
                    f"{confidence:.0f}% detection confidence"
                ):

                    st.write(
                        f"**Claim:** {claim['claim']}"
                    )

                    st.progress(
                        int(
                            max(
                                0,
                                min(
                                    100,
                                    round(
                                        confidence
                                    )
                                )
                            )
                        )
                    )

                    if reasons:

                        st.write(
                            "**Detection evidence:** "
                            + ", ".join(reasons)
                        )

                    if claim.get("cue_words"):

                        st.write(
                            "**Cue words:** "
                            + ", ".join(
                                claim["cue_words"]
                            )
                        )

                    if claim.get("numbers"):

                        st.write(
                            "**Quantitative information:** "
                            + ", ".join(
                                claim["numbers"]
                            )
                        )

                    if claim.get("entities"):

                        st.write(
                            "**Named entities:** "
                            + ", ".join(
                                claim["entities"]
                            )
                        )

        else:

            st.success(
                "🟢 No strong factual-claim candidates "
                "were detected in this article."
            )

        st.caption(
            "⚠️ Claim Detection identifies statements that may be "
            "factually verifiable. It does not verify whether the "
            "claim is true or false. Verification will be handled "
            "by a future fact-checking/verification module."
        )

        # ====================================================
        # CONTRADICTION DETECTION
        # ====================================================

        st.divider()

        st.header(
            "⚠️ Contradiction Detection"
        )

        st.write(
            "Compare important statements in the article and "
            "identify pairs that may contradict each other."
        )

        with st.spinner(
            "Checking statements for contradictions..."
        ):

            contradiction_result = detect_contradictions(
                processed_text
            )

        contradiction_count = int(
            contradiction_result.get(
                "contradiction_count",
                0
            )
        )

        contradiction_score = float(
            contradiction_result.get(
                "contradiction_score",
                0.0
            )
        )

        contradiction_level = contradiction_result.get(
            "level",
            "No Contradictions Detected"
        )

        pairs_checked = int(
            contradiction_result.get(
                "pairs_checked",
                0
            )
        )

        contradiction_items = contradiction_result.get(
            "contradictions",
            []
        )

        contradiction_col1, contradiction_col2, contradiction_col3 = st.columns(3)

        with contradiction_col1:

            st.metric(
                "Contradictions",
                contradiction_count
            )

        with contradiction_col2:

            st.metric(
                "Pairs Checked",
                pairs_checked
            )

        with contradiction_col3:

            st.metric(
                "Contradiction Level",
                contradiction_level
            )

        if contradiction_items:

            st.warning(
                f"⚠️ {contradiction_count} potentially "
                "contradictory statement pair(s) detected."
            )

            for index, item in enumerate(
                contradiction_items,
                start=1
            ):

                confidence = item.get(
                    "confidence",
                    0.0
                )

                with st.expander(
                    f"Contradiction {index} — "
                    f"{confidence:.2f}% confidence"
                ):

                    st.write(
                        "**Statement 1:**"
                    )

                    st.info(
                        item["statement_1"]
                    )

                    st.write(
                        "**Statement 2:**"
                    )

                    st.info(
                        item["statement_2"]
                    )

                    st.progress(
                        int(
                            max(
                                0,
                                min(
                                    100,
                                    round(
                                        confidence
                                    )
                                )
                            )
                        )
                    )

                    st.caption(
                        f"Contradiction confidence: "
                        f"{confidence:.2f}%"
                    )

        else:

            st.success(
                "🟢 No strong internal contradictions "
                "were detected."
            )

        st.caption(
            "⚠️ Contradiction Detection uses a Natural Language "
            "Inference model to identify statements that may conflict. "
            "A detected contradiction is a warning for review, not "
            "proof that either statement is false."
        )

        # ====================================================
        # FAKE NEWS DETECTION
        # ====================================================

        st.divider()

        st.header(
            "🕵️ Fake News Detection"
        )

        with st.spinner(
            "Checking news authenticity..."
        ):

            fake_prediction, fake_confidence = detect_fake_news(
                processed_text
            )

        col1, col2 = st.columns(2)

        with col1:

            if fake_prediction == "LIKELY REAL":

                st.success(
                    f"🟢 {fake_prediction}"
                )

            elif fake_prediction == "POTENTIALLY FAKE":

                st.warning(
                    f"⚠️ {fake_prediction}"
                )

            else:

                st.info(
                    f"ℹ️ {fake_prediction}"
                )

        with col2:

            st.metric(
                "Model Confidence",
                f"{fake_confidence * 100:.2f}%"
            )

        st.caption(
            "This prediction indicates how closely the article "
            "matches patterns learned from the fake/real news "
            "training dataset. It does not prove whether the "
            "article is factually true or false."
        )

        # ====================================================
        # NEWS CREDIBILITY SCORE
        # ====================================================

        st.divider()

        st.header(
            "🧠 News Credibility Score"
        )

        st.write(
            "This score combines the current fake-news prediction "
            "with sentiment indicators. Bias and claim evidence "
            "will be incorporated more strongly as those modules "
            "are added."
        )

        with st.spinner(
            "Calculating credibility score..."
        ):

            credibility_score = calculate_credibility_score(
                fake_confidence=fake_confidence,
                fake_prediction=fake_prediction,
                sentiment_score=sentiment_score,
                bias_score=bias_score,
                claim_score=claim_score
            )

            # Contradictions reduce credibility as an additional
            # consistency penalty. This is intentionally capped.
            contradiction_penalty = min(
                20.0,
                contradiction_score * 0.20
            )

            credibility_score = round(
                max(
                    0.0,
                    credibility_score - contradiction_penalty
                ),
                2
            )

            credibility_level = get_credibility_level(
                credibility_score
            )

        credibility_col1, credibility_col2, credibility_col3 = st.columns(3)

        with credibility_col1:
            st.metric(
                "Credibility Score",
                f"{credibility_score:.2f}/100"
            )

        with credibility_col2:
            st.metric(
                "Credibility Level",
                credibility_level
            )

        with credibility_col3:
            if credibility_score >= 80:
                status_text = "🟢 Strong"
            elif credibility_score >= 60:
                status_text = "🟡 Moderate"
            elif credibility_score >= 40:
                status_text = "🟠 Low"
            else:
                status_text = "🔴 Very Low"

            st.metric(
                "Overall Indicator",
                status_text
            )

        st.progress(
            int(max(0, min(100, round(credibility_score))))
        )

        # Explain the score.
        if credibility_score >= 80:
            st.success(
                "🟢 The article shows strong credibility indicators "
                "according to the current AI-based analysis."
            )
        elif credibility_score >= 60:
            st.warning(
                "🟡 The article shows moderate credibility indicators. "
                "Further verification is recommended."
            )
        elif credibility_score >= 40:
            st.warning(
                "🟠 The article shows relatively low credibility "
                "indicators. Verify important claims using reliable sources."
            )
        else:
            st.error(
                "🔴 The article shows very low credibility indicators. "
                "Do not treat this score as proof that the article is false."
            )

        st.info(
            "⚠️ The credibility score is an AI-based indicator, "
            "not a factual truth certificate. A high score does not "
            "guarantee that an article is true."
        )

        # ====================================================
        # EXPLAINABLE AI
        # ====================================================

        st.divider()

        st.header(
            "🧠 Explainable AI"
        )

        st.write(
            f"Why did the model classify this article "
            f"as **{top_category['category']}**?"
        )

        try:
            with st.spinner("Generating explanation (LIME)..."):
                explanation = explain_prediction(processed_text, num_samples=80)
        except Exception as exc:
            explanation = []
            st.error(f"Explainable AI failed: {type(exc).__name__}: {exc}")

        if explanation:

            explanation_df = pd.DataFrame(
                explanation,
                columns=[
                    "Word/Phrase",
                    "Importance"
                ]
            )

            explanation_df["Importance"] = pd.to_numeric(
                explanation_df["Importance"],
                errors="coerce"
            )

            explanation_df = explanation_df.dropna(
                subset=["Importance"]
            )

            if not explanation_df.empty:

                explanation_df[
                    "Absolute Importance"
                ] = explanation_df[
                    "Importance"
                ].abs()

                explanation_df = (
                    explanation_df
                    .sort_values(
                        "Absolute Importance",
                        ascending=False
                    )
                    .drop(
                        columns=[
                            "Absolute Importance"
                        ]
                    )
                )

                st.subheader(
                    "🔍 Feature Importance"
                )

                st.dataframe(
                    explanation_df,
                    width="stretch"
                )

                st.subheader(
                    "📊 Important Words / Phrases"
                )

                chart_df = (
                    explanation_df
                    .head(15)
                    .copy()
                )

                chart_df = chart_df.sort_values(
                    "Importance"
                )

                fig2 = px.bar(
                    chart_df,
                    x="Importance",
                    y="Word/Phrase",
                    orientation="h",
                    title="Explainable AI - Feature Importance",
                    labels={
                        "Importance": "Importance",
                        "Word/Phrase": "Word / Phrase"
                    }
                )

                st.plotly_chart(
                    fig2,
                    width="stretch"
                )

                st.caption(
                    "Positive values indicate features that "
                    "support the prediction, while negative "
                    "values indicate features that push the "
                    "prediction away from it."
                )

            else:

                st.info(
                    "No explanation was generated."
                )

        else:

            st.warning(
                "No explanation weights were returned. Check the terminal for the detailed diagnostic. "
                "The analyzer will continue to show the other results; this message does not mean the article is fake or incorrect."
            )

        # ====================================================
        # SIMILAR NEWS DETECTION
        # ====================================================

        st.divider()

        st.header(
            "🔎 Similar News Detection"
        )

        st.write(
            "Find previously analyzed articles that are meaningfully related. "
            "Exact and near-duplicate copies are excluded so the list is more useful."
        )

        with st.spinner(
            "Searching for previous news..."
        ):

            previous_news = get_all_news()

        if (
            isinstance(previous_news, pd.DataFrame)
            and not previous_news.empty
        ):

            news_data = previous_news.rename(
                columns={
                    "article_text": "text"
                }
            )

            # Remove empty articles
            news_data = news_data[
                news_data["text"].fillna("").str.strip() != ""
            ]

            if not news_data.empty:

                with st.spinner(
                    "Comparing article meaning..."
                ):

                    similar_articles = find_similar_news(
                        processed_text,
                        news_data,
                        top_n=5
                    )

            else:

                similar_articles = pd.DataFrame()

        else:

            similar_articles = pd.DataFrame()

        if not similar_articles.empty:

            st.subheader(
                "📰 Related Articles"
            )

            for _, article in similar_articles.iterrows():

                similarity = (
                    article["similarity"] * 100
                )

                st.markdown(
                    f"### 📰 {article['title']}"
                )

                st.progress(
                    min(
                        max(int(similarity), 0),
                        100
                    )
                )

                st.write(
                    f"**Similarity: {similarity:.2f}%**"
                )

                st.caption(
                    f"Language: {article['language']}  |  "
                    f"Analyzed: {article['created_at']}"
                )

                st.write(
                    article["text"]
                )

                st.divider()

        else:

            st.info(
                "No previous articles are available "
                "for similarity comparison."
            )

        # ====================================================
        # SAVE ANALYSIS TO DATABASE
        # ====================================================

        if st.session_state.user:

            user_id = st.session_state.user["id"]

            # ------------------------------------------------
            # SAVE ORIGINAL ARTICLE
            # ------------------------------------------------

            news_id = save_news(
                user_id=user_id,
                title=f"{top_category['category']} News",
                article_text=text,
                language=detected_language
            )

            # ------------------------------------------------
            # SAVE ANALYSIS RESULTS
            # ------------------------------------------------

            # Persist an explicit review status when this analysis has a
            # concrete review trigger. Keep the live model prediction shown
            # above unchanged; only the saved status used by the dashboard
            # is changed so the pending-review metric can count it.
            review_required = (
                str(fake_prediction).strip().casefold() == "potentially fake"
                or contradiction_count > 0
                or credibility_score < 40
            )
            saved_review_status = "NEEDS REVIEW" if review_required else fake_prediction

            save_analysis(
                news_id=news_id,
                category=top_category["category"],
                sentiment=sentiment_label,
                sentiment_score=sentiment_score,
                summary=summary,
                fake_prediction=saved_review_status,
                fake_confidence=fake_confidence
            )

            st.success(
                "💾 Analysis saved successfully!"
            )


# ============================================================
# PROFILE PAGE
# ============================================================

def profile_page():

    set_background(
        "news_background.jpg"
    )

    sidebar()

    st.title(
        "👤 My Profile"
    )

    user = st.session_state.user

    if user:

        st.write(
            f"### 👤 Name: {user['name']}"
        )

        st.write(
            f"### 📱 Phone: {user['phone']}"
        )

        st.divider()

        st.info(
            """
            🔐 Your password is securely stored as a
            cryptographic hash and is never displayed.
            """
        )

        st.success(
            "Your account is protected."
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

if not st.session_state.logged_in:

    if st.session_state.page == "Sign Up":

        signup_page()

    else:

        login_page()

else:

    if st.session_state.page == "Analyzer":
        analyzer_page()
    elif st.session_state.page == "Profile":
        profile_page()
    elif st.session_state.page == "History":
        history_page()
    elif st.session_state.page == "Connections":
        connections_page()
    elif st.session_state.page == "Claim Explorer":
        claim_explorer_page()
    else:
        dashboard()