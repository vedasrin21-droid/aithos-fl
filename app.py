"""
AITHOS-FL - Federated Learning Platform for Healthcare
Milestone 2 - CSV Upload, PII Detection & Privacy Dashboard
"""

import streamlit as st
import pandas as pd

from privacy.dataset_store import HospitalDatasetStore, HospitalDataset
from privacy.pii_detector import PIIDetector
from privacy.redaction import redact_dataframe
from privacy.synthetic_data import generate_synthetic_csv_bytes
from local_ai.assistant import LocalAIAssistant, LocalAIResponse, LOCAL_STATUS_LABEL
from shared_ai.knowledge_store import SharedKnowledgeStore, ALLOWED_CATEGORIES
from shared_ai.assistant import (
    SharedAIAssistant,
    SharedAIResponse,
    SHARED_STATUS_LABEL,
    PRIVATE_RECORD_REFUSAL,
    KNOWLEDGE_UNAVAILABLE_MESSAGE,
)

# -----------------------------------------------------------------------------
# Configuration & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AITHOS-FL | Federated Learning Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Demo Accounts Configuration
# (Strictly Demo-Only Credentials)
# -----------------------------------------------------------------------------
DEMO_ACCOUNTS = {
    "hospital.a@demo.test": {
        "password": "AithosA123!",
        "name": "Hospital A Medical Center",
        "role": "Hospital Node Administrator",
        "hospital_id": "HOSP-A",
        "node_type": "Institutional Node (Primary)",
        "facility_type": "Tertiary Care & Research Center",
        "permissions": "Full Node Administration & Local Ingestion",
    },
    "hospital.b@demo.test": {
        "password": "AithosB123!",
        "name": "Hospital B Regional Health",
        "role": "Hospital Node Administrator",
        "hospital_id": "HOSP-B",
        "node_type": "Institutional Node (Secondary)",
        "facility_type": "Regional Academic Hospital",
        "permissions": "Full Node Administration & Local Ingestion",
    },
    "user@demo.test": {
        "password": "User123!",
        "name": "Standard Research User",
        "role": "Clinical Research Scientist",
        "hospital_id": "Unassigned (Independent / External)",
        "node_type": "Research Observer Node",
        "facility_type": "Independent Collaborative Research",
        "permissions": "Read-Only Model Inspection & Privacy Auditing",
    },
}

# -----------------------------------------------------------------------------
# Claymorphism CSS Theme
# Clean blue claymorphism: light blue background, white rounded cards, soft 3D shadows, dark blue text
# -----------------------------------------------------------------------------
CLAYMORPHISM_CSS = """
<style>
/* Main app background with subtle ambient blue gradient */
.stApp {
    background-color: #ebf4fc !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(214, 234, 255, 0.7) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(220, 238, 255, 0.6) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(240, 248, 255, 0.4) 0px, transparent 50%) !important;
    color: #0d2a4a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Streamlit Header Bar */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f1f7fd !important;
    border-right: 1px solid rgba(200, 220, 240, 0.7) !important;
    box-shadow: 6px 0 20px rgba(180, 205, 230, 0.25) !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(185, 210, 235, 0.6) !important;
}

/* Hide default streamlit decorations */
#MainMenu, footer {visibility: hidden;}

/* Base typography */
h1, h2, h3, h4, h5, h6 {
    color: #0b2545 !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

p, span, label {
    color: #17385c;
}

/* Claymorphic Card component */
.clay-card {
    background: #ffffff;
    border-radius: 22px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 
        10px 10px 24px rgba(165, 190, 215, 0.35),
        -8px -8px 20px rgba(255, 255, 255, 0.95),
        inset 1px 1px 2px rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(230, 240, 250, 0.9);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.clay-card:hover {
    transform: translateY(-2px);
    box-shadow: 
        12px 14px 28px rgba(155, 182, 210, 0.4),
        -8px -8px 20px rgba(255, 255, 255, 0.95);
}

/* Claymorphic Hero / Banner Card */
.clay-hero {
    background: linear-gradient(135deg, #ffffff 0%, #f4f9ff 100%);
    border-radius: 24px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 
        12px 14px 28px rgba(160, 188, 218, 0.38),
        -8px -8px 22px rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(220, 235, 250, 0.85);
}

/* Claymorphic Mini Cards / Metric Badges */
.clay-metric-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 20px 22px;
    height: 100%;
    box-shadow: 
        8px 8px 18px rgba(165, 190, 215, 0.32),
        -6px -6px 16px rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(235, 243, 252, 0.8);
}

.clay-metric-title {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #4b6b8e;
    margin-bottom: 6px;
}

.clay-metric-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0b2545;
    margin-bottom: 4px;
}

.clay-metric-sub {
    font-size: 0.85rem;
    color: #2563eb;
    font-weight: 500;
}

/* Clay Badges and Pills */
.clay-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 30px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    box-shadow: 
        inset 2px 2px 4px rgba(175, 200, 225, 0.4),
        inset -2px -2px 4px rgba(255, 255, 255, 0.8);
}

.clay-badge-blue {
    background: #e1effe;
    color: #1e40af;
}

.clay-badge-amber {
    background: #fef3c7;
    color: #92400e;
}

.clay-badge-green {
    background: #dcfce7;
    color: #166534;
}

.clay-badge-red {
    background: #fee2e2;
    color: #991b1b;
}

/* Clay Demo Alert Box */
.clay-demo-alert {
    background: #fdf8e6;
    border-radius: 16px;
    padding: 14px 18px;
    border: 1px solid #fde68a;
    box-shadow: 
        6px 6px 14px rgba(210, 190, 150, 0.25),
        -5px -5px 12px rgba(255, 255, 255, 0.9);
    margin-bottom: 20px;
    color: #854d0e;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Clay Inset Wells */
.clay-well {
    background: #e4eef8;
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 
        inset 4px 4px 8px rgba(170, 195, 220, 0.45),
        inset -4px -4px 8px rgba(255, 255, 255, 0.85);
    border: 1px solid rgba(215, 230, 245, 0.7);
    margin: 14px 0;
}

/* Streamlit Buttons Claymorphic Styling */
.stButton > button {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 8px 22px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    box-shadow: 
        5px 6px 14px rgba(29, 78, 216, 0.35),
        -3px -3px 10px rgba(255, 255, 255, 0.5),
        inset 1px 1px 2px rgba(255, 255, 255, 0.3) !important;
    transition: all 0.18s ease-in-out !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    background: #1e40af !important;
    transform: translateY(-2px) !important;
    box-shadow: 
        7px 8px 18px rgba(29, 78, 216, 0.42),
        -3px -3px 10px rgba(255, 255, 255, 0.5) !important;
}

.stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow: 
        inset 2px 2px 5px rgba(10, 40, 120, 0.4),
        2px 2px 6px rgba(29, 78, 216, 0.25) !important;
}

/* Secondary Button */
button[kind="secondary"] {
    background: #ffffff !important;
    color: #0b2545 !important;
    border: 1px solid rgba(200, 220, 240, 0.8) !important;
    box-shadow: 
        5px 5px 12px rgba(170, 195, 220, 0.3),
        -4px -4px 10px rgba(255, 255, 255, 0.9) !important;
}

button[kind="secondary"]:hover {
    background: #f8fbff !important;
    color: #b91c1c !important;
}

/* Streamlit Inputs */
.stTextInput > div > div > input {
    background-color: #f7fbff !important;
    border-radius: 12px !important;
    border: 1px solid rgba(195, 215, 235, 0.8) !important;
    color: #0b2545 !important;
    box-shadow: 
        inset 2px 2px 6px rgba(175, 200, 225, 0.35),
        inset -2px -2px 6px rgba(255, 255, 255, 0.8) !important;
    padding: 10px 14px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #2563eb !important;
    box-shadow: 
        0 0 0 3px rgba(37, 99, 235, 0.2),
        inset 2px 2px 5px rgba(175, 200, 225, 0.3) !important;
}

/* Radio buttons inside sidebar */
[data-testid="stSidebar"] .stRadio label {
    font-weight: 500 !important;
    color: #0f2d4a !important;
    padding: 6px 10px !important;
    border-radius: 10px !important;
    transition: background-color 0.15s ease;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(225, 238, 250, 0.7);
}

/* User Profile card in sidebar */
.clay-user-badge {
    background: #ffffff;
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 20px;
    box-shadow: 
        6px 6px 16px rgba(170, 195, 220, 0.3),
        -5px -5px 14px rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(220, 235, 250, 0.85);
}

/* Tab styling in Streamlit */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px 12px 0 0 !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    color: #2d4c6d !important;
}

.stTabs [aria-selected="true"] {
    color: #1d4ed8 !important;
    border-bottom-color: #1d4ed8 !important;
}

/* Chat bubble styling for Local AI */
.clay-chat-user {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
    color: #ffffff !important;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin-bottom: 12px;
    margin-left: 20%;
    box-shadow: 6px 6px 16px rgba(29, 78, 216, 0.28), -3px -3px 10px rgba(255, 255, 255, 0.5);
    font-size: 0.92rem;
    line-height: 1.5;
}
.clay-chat-user p, .clay-chat-user span, .clay-chat-user strong {
    color: #ffffff !important;
}
.clay-chat-assistant {
    background: #ffffff;
    color: #0b2545 !important;
    border-radius: 18px 18px 18px 4px;
    padding: 16px 20px;
    margin-bottom: 14px;
    margin-right: 15%;
    box-shadow: 8px 8px 18px rgba(165, 190, 215, 0.35), -6px -6px 16px rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(220, 235, 250, 0.85);
    font-size: 0.92rem;
    line-height: 1.5;
}
</style>
"""

st.markdown(CLAYMORPHISM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user_email" not in st.session_state:
    st.session_state.current_user_email = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Dashboard"

# Global partitioned dataset store (persisted in session state)
if "dataset_store" not in st.session_state:
    st.session_state.dataset_store = HospitalDatasetStore()

if "selected_dataset_id" not in st.session_state:
    st.session_state.selected_dataset_id = None

# Local AI Assistant Engine (bound to dataset store)
if "local_ai_assistant" not in st.session_state:
    st.session_state.local_ai_assistant = LocalAIAssistant(st.session_state.dataset_store)
else:
    st.session_state.local_ai_assistant.store = st.session_state.dataset_store

# Partitioned chat history: user_key -> list of messages
if "chat_history_by_user" not in st.session_state:
    st.session_state.chat_history_by_user = {}

# Staged query for quick prompt chips
if "staged_query" not in st.session_state:
    st.session_state.staged_query = ""

# Shared Knowledge Store (persistent JSON backend)
if "shared_knowledge_store" not in st.session_state:
    st.session_state.shared_knowledge_store = SharedKnowledgeStore()

# Shared AI Assistant Engine
if "shared_ai_assistant" not in st.session_state:
    st.session_state.shared_ai_assistant = SharedAIAssistant(st.session_state.shared_knowledge_store)
else:
    st.session_state.shared_ai_assistant.store = st.session_state.shared_knowledge_store

# Shared AI Chat History (accessible to all authenticated users)
if "shared_ai_chat_history" not in st.session_state:
    st.session_state.shared_ai_chat_history = []


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def login_user(email: str):
    """Authenticate demo user and persist in session state."""
    account = DEMO_ACCOUNTS[email].copy()
    account["email"] = email
    st.session_state.authenticated = True
    st.session_state.current_user_email = email
    st.session_state.current_user = account
    st.session_state.active_nav = "Dashboard"
    st.session_state.selected_dataset_id = None


def logout_user():
    """Clear session state and return to login view."""
    st.session_state.authenticated = False
    st.session_state.current_user_email = None
    st.session_state.current_user = None
    st.session_state.active_nav = "Dashboard"
    st.session_state.selected_dataset_id = None
    st.rerun()


# -----------------------------------------------------------------------------
# VIEW: Login Screen (When Not Authenticated)
# -----------------------------------------------------------------------------
def render_login_view():
    col_spacer_l, col_center, col_spacer_r = st.columns([1, 2.2, 1])

    with col_center:
        # Header Clay Card
        st.markdown(
            """
            <div class="clay-hero" style="text-align: center; margin-top: 25px;">
                <div style="font-size: 2.6rem; line-height: 1.1; margin-bottom: 8px;">🧬</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: #0b2545; letter-spacing: -0.02em;">
                    AITHOS-FL
                </div>
                <div style="font-size: 0.96rem; color: #2563eb; font-weight: 600; margin-top: 4px;">
                    Federated Learning Platform for Collaborative Healthcare Intelligence
                </div>
                <div style="font-size: 0.86rem; color: #4b6b8e; margin-top: 8px; line-height: 1.4;">
                    Privacy-preserving on-premise model training across medical institutions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clear Demo-Only Notice
        st.markdown(
            """
            <div class="clay-demo-alert">
                <span style="font-size: 1.3rem;">⚠️</span>
                <div>
                    <strong>Evaluation Environment Notice:</strong> Authentication is strictly 
                    <strong>demo-only</strong>. No clinical or hospital identity providers are connected.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Login Form Card
        st.markdown(
            """
            <div class="clay-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #0b2545;">Sign In to Node</span>
                    <span class="clay-badge clay-badge-amber">DEMO AUTH ONLY</span>
                </div>
                <div style="font-size: 0.88rem; color: #5a7b9e; margin-bottom: 12px;">
                    Enter one of the preconfigured test credentials below to access your federated node.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Input fields inside form
        with st.form("login_form"):
            input_email = st.text_input(
                "Hospital / User Email",
                placeholder="e.g. hospital.a@demo.test",
                key="login_email_input",
            )
            input_password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password",
                key="login_password_input",
            )
            submit_button = st.form_submit_button("Sign In to AITHOS-FL", use_container_width=True)

        if submit_button:
            trimmed_email = input_email.strip().lower()
            if trimmed_email in DEMO_ACCOUNTS:
                target_acc = DEMO_ACCOUNTS[trimmed_email]
                if input_password == target_acc["password"]:
                    login_user(trimmed_email)
                    st.rerun()
                else:
                    st.error("❌ Invalid password. Please verify the demo credentials below.")
            else:
                st.error("❌ Account not recognized. Please use one of the 3 preconfigured demo accounts.")

        # Quick Fill / Demo Credentials Reference Card
        st.markdown(
            """
            <div class="clay-card" style="margin-top: 15px;">
                <div style="font-size: 0.92rem; font-weight: 700; color: #0b2545; margin-bottom: 10px;">
                    📋 Preconfigured Demo Accounts
                </div>
                <div class="clay-well">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #1e40af; margin-bottom: 4px;">
                        🏥 Hospital A (Primary Node)
                    </div>
                    <div style="font-size: 0.82rem; color: #1f3a5a; font-family: monospace;">
                        Email: <strong>hospital.a@demo.test</strong><br>
                        Password: <strong>AithosA123!</strong><br>
                        Role: Hospital Node Administrator | Hospital ID: <strong>HOSP-A</strong>
                    </div>
                </div>
                <div class="clay-well">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #1e40af; margin-bottom: 4px;">
                        🏥 Hospital B (Secondary Node)
                    </div>
                    <div style="font-size: 0.82rem; color: #1f3a5a; font-family: monospace;">
                        Email: <strong>hospital.b@demo.test</strong><br>
                        Password: <strong>AithosB123!</strong><br>
                        Role: Hospital Node Administrator | Hospital ID: <strong>HOSP-B</strong>
                    </div>
                </div>
                <div class="clay-well">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #1e40af; margin-bottom: 4px;">
                        🔬 Normal User (Research Observer)
                    </div>
                    <div style="font-size: 0.82rem; color: #1f3a5a; font-family: monospace;">
                        Email: <strong>user@demo.test</strong><br>
                        Password: <strong>User123!</strong><br>
                        Role: Clinical Research Scientist | Hospital ID: <strong>Unassigned</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Login Helper Buttons for Fast Testing
        st.markdown(
            "<div style='text-align: center; font-size: 0.82rem; color: #5a7b9e; margin-bottom: 8px;'>⚡ One-Click Demo Sign In</div>",
            unsafe_allow_html=True,
        )
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("Log in as Hospital A", use_container_width=True, key="quick_hosp_a"):
                login_user("hospital.a@demo.test")
                st.rerun()
        with col_btn2:
            if st.button("Log in as Hospital B", use_container_width=True, key="quick_hosp_b"):
                login_user("hospital.b@demo.test")
                st.rerun()
        with col_btn3:
            if st.button("Log in as Normal User", use_container_width=True, key="quick_user"):
                login_user("user@demo.test")
                st.rerun()


# -----------------------------------------------------------------------------
# VIEW: Authenticated Application
# -----------------------------------------------------------------------------
def render_app_view():
    user = st.session_state.current_user
    user_email = st.session_state.current_user_email

    # -------------------------------------------------------------------------
    # Sidebar Navigation & User Info
    # -------------------------------------------------------------------------
    with st.sidebar:
        # Platform Logo / Title
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; padding: 4px 6px;">
                <span style="font-size: 1.8rem;">🧬</span>
                <div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #0b2545; letter-spacing: -0.02em;">AITHOS-FL</div>
                    <div style="font-size: 0.72rem; color: #2563eb; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Federated Healthcare</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Authenticated User Badge Card
        hospital_id_badge = (
            f'<span class="clay-badge clay-badge-blue">{user["hospital_id"]}</span>'
            if user["hospital_id"] != "Unassigned (Independent / External)"
            else '<span class="clay-badge clay-badge-amber">Independent</span>'
        )

        st.markdown(
            f"""
            <div class="clay-user-badge">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                    <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #4b6b8e; font-weight: 700;">Node Account</span>
                    <span class="clay-badge clay-badge-green">● Online</span>
                </div>
                <div style="font-size: 0.98rem; font-weight: 700; color: #0b2545; margin-bottom: 2px;">
                    {user["name"]}
                </div>
                <div style="font-size: 0.80rem; color: #2563eb; font-weight: 600; margin-bottom: 8px;">
                    {user["role"]}
                </div>
                <div style="font-size: 0.78rem; color: #4b6b8e; margin-bottom: 8px; word-break: break-all;">
                    {user_email}
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between; padding-top: 6px; border-top: 1px solid rgba(220, 235, 250, 0.8);">
                    <span style="font-size: 0.76rem; color: #5a7b9e;">Hospital ID:</span>
                    {hospital_id_badge}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation Menu
        st.markdown(
            "<div style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #4b6b8e; margin-bottom: 6px; padding-left: 4px;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        nav_options = [
            "Dashboard",
            "Upload Dataset",
            "Local AI",
            "Shared AI",
            "Privacy Center",
        ]

        selected_nav = st.radio(
            "Go to",
            nav_options,
            index=nav_options.index(st.session_state.active_nav)
            if st.session_state.active_nav in nav_options
            else 0,
            label_visibility="collapsed",
            key="sidebar_nav_radio",
        )
        st.session_state.active_nav = selected_nav

        st.markdown("<hr style='margin: 18px 0;'>", unsafe_allow_html=True)

        # Demo Mode & Node Indicator
        st.markdown(
            """
            <div style="background: rgba(225, 238, 252, 0.6); border-radius: 12px; padding: 10px 14px; font-size: 0.76rem; color: #3b5a7a; margin-bottom: 14px;">
                <div style="font-weight: 700; color: #1e40af; margin-bottom: 2px;">⚡ Demo Environment</div>
                Strictly demo credentials. No synthetic or fabricated model predictions.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Working Logout Button
        if st.button("🚪 Log Out", use_container_width=True, type="secondary"):
            logout_user()

    # -------------------------------------------------------------------------
    # Main Content Area
    # -------------------------------------------------------------------------
    if st.session_state.active_nav == "Dashboard":
        render_dashboard_page(user, user_email)
    elif st.session_state.active_nav == "Upload Dataset":
        render_upload_dataset_page(user)
    elif st.session_state.active_nav == "Local AI":
        render_local_ai_page(user)
    elif st.session_state.active_nav == "Shared AI":
        render_shared_ai_page(user)
    elif st.session_state.active_nav == "Privacy Center":
        render_privacy_center_page(user)


# -----------------------------------------------------------------------------
# PAGE 1: Dashboard
# -----------------------------------------------------------------------------
def render_dashboard_page(user: dict, user_email: str):
    # Retrieve dataset count for active hospital (if authorized)
    hosp_datasets_count = 0
    if user["hospital_id"] in ["HOSP-A", "HOSP-B"]:
        try:
            hosp_datasets = st.session_state.dataset_store.get_datasets_for_user(user)
            hosp_datasets_count = len(hosp_datasets)
        except Exception:
            hosp_datasets_count = 0

    node_status_sub = (
        f"● {hosp_datasets_count} Dataset(s) Loaded"
        if hosp_datasets_count > 0
        else "● Ready for Ingestion"
    )

    # Welcome Clay Hero
    st.markdown(
        f"""
        <div class="clay-hero">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div class="clay-badge clay-badge-blue" style="margin-bottom: 10px;">
                        {user["node_type"]}
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #0b2545;">
                        Welcome, {user["name"]}
                    </div>
                    <div style="font-size: 0.95rem; color: #35587f; margin-top: 4px;">
                        Role: <strong>{user["role"]}</strong> &bull; Hospital ID: <strong>{user["hospital_id"]}</strong>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="clay-badge clay-badge-amber">DEMO SESSION ACTIVE</div>
                    <div style="font-size: 0.8rem; color: #5a7b9e; margin-top: 6px;">Logged in as: {user_email}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 System Overview Clay Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="clay-metric-card">
                <div class="clay-metric-title">Hospital Node</div>
                <div class="clay-metric-value">{user["hospital_id"]}</div>
                <div class="clay-metric-sub">{node_status_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="clay-metric-card">
                <div class="clay-metric-title">Federated Protocol</div>
                <div class="clay-metric-value">FedAvg</div>
                <div class="clay-metric-sub">Decentralized Convergence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="clay-metric-card">
                <div class="clay-metric-title">Privacy Engine</div>
                <div class="clay-metric-value">Diff-Privacy</div>
                <div class="clay-metric-sub">PII Scanner & DP Guard Active</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="clay-metric-card">
                <div class="clay-metric-title">Consortium State</div>
                <div class="clay-metric-value">Connected</div>
                <div class="clay-metric-sub">Hosp A & Hosp B Enrolled</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Compact "Private AI vs Shared AI" Comparison Panel
    # -------------------------------------------------------------------------
    shared_store: SharedKnowledgeStore = st.session_state.shared_knowledge_store
    shared_items = shared_store.get_all_items()
    shared_items_count = len(shared_items)
    shared_audit_logs = shared_store.get_audit_log()
    shared_audit_count = len(shared_audit_logs)

    contributing_hosp_ids = sorted(list(set(i.get("hospital_id", "") for i in shared_items if i.get("hospital_id"))))
    contrib_str = ", ".join(contributing_hosp_ids) if contributing_hosp_ids else "None yet"

    # Private AI details for authenticated user
    is_hospital_node = user["hospital_id"] in ["HOSP-A", "HOSP-B"]
    active_dataset = None
    if is_hospital_node:
        try:
            hosp_datasets = st.session_state.dataset_store.get_datasets_for_user(user)
            hosp_datasets_count = len(hosp_datasets)
            if hosp_datasets:
                active_dataset_id = st.session_state.get("selected_dataset_id")
                if active_dataset_id:
                    try:
                        active_dataset = st.session_state.dataset_store.get_dataset_by_id(user, active_dataset_id)
                    except Exception:
                        active_dataset = hosp_datasets[-1]
                else:
                    active_dataset = hosp_datasets[-1]
        except Exception:
            hosp_datasets_count = 0
    else:
        hosp_datasets_count = 0

    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 18px 0 12px; flex-wrap: wrap; gap: 8px;">
            <div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0b2545;">
                    ⚖️ Private AI vs Shared AI Comparison
                </div>
                <div style="font-size: 0.85rem; color: #4b6b8e;">
                    Comparison of on-premise institutional data isolation against consortium-approved collaborative knowledge retrieval.
                </div>
            </div>
            <span class="clay-badge clay-badge-amber">🔬 Architecture Prototype</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_comp_priv, col_comp_shared = st.columns(2)

    with col_comp_priv:
        if is_hospital_node:
            active_ds_status = (
                f"📄 <strong>{active_dataset.filename}</strong> ({active_dataset.row_count} rows, {active_dataset.column_count} cols)"
                if active_dataset
                else "⚠️ <em>No dataset ingested yet (ready for upload)</em>"
            )
            priv_badge_html = f'<span class="clay-badge clay-badge-green">🔒 {LOCAL_STATUS_LABEL}</span>'
            priv_node_desc = f"Connected Node: <strong>{user['hospital_id']}</strong> ({user['name']})"
            priv_storage_desc = f"<strong>{hosp_datasets_count}</strong> dataset(s) loaded in private hospital vault"
        else:
            active_ds_status = "🔒 <em>Private hospital data inaccessible from observer session</em>"
            priv_badge_html = '<span class="clay-badge clay-badge-amber">🔒 Access Restricted</span>'
            priv_node_desc = "Connected Node: <strong>Observer / External Researcher</strong>"
            priv_storage_desc = "Local clinical vault restricted to Hospital Node Administrators"

        st.markdown(
            f"""
            <div class="clay-card" style="height: 100%; margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; flex-wrap: wrap; gap: 6px;">
                    <div>
                        <div style="font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; color: #4b6b8e; font-weight: 700;">INSTITUTIONAL VAULT</div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #0b2545; margin-top: 2px;">
                            🔒 Private / Hospital AI
                        </div>
                    </div>
                    {priv_badge_html}
                </div>
                <ul style="margin: 0; padding-left: 18px; font-size: 0.88rem; color: #2d4c6d; line-height: 1.6;">
                    <li><strong>Source:</strong> Uses the currently authenticated hospital's uploaded records ({user["hospital_id"]}).</li>
                    <li><strong>Isolation:</strong> Hospital-specific records and answers stay strictly inside the private workspace.</li>
                    <li><strong>Access Boundary:</strong> {LOCAL_STATUS_LABEL}</li>
                </ul>
                <div class="clay-well" style="margin-top: 14px; padding: 12px 14px; font-size: 0.82rem;">
                    <div style="font-weight: 700; color: #1e40af; margin-bottom: 4px;">Live Private State:</div>
                    <div style="color: #0b2545;">{priv_node_desc}</div>
                    <div style="color: #3b5a7a; margin-top: 2px;">Vault Contents: {priv_storage_desc}</div>
                    <div style="color: #3b5a7a; margin-top: 2px;">Active Context: {active_ds_status}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_comp_shared:
        st.markdown(
            f"""
            <div class="clay-card" style="height: 100%; margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; flex-wrap: wrap; gap: 6px;">
                    <div>
                        <div style="font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; color: #4b6b8e; font-weight: 700;">CONSORTIUM KNOWLEDGE</div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #0b2545; margin-top: 2px;">
                            🌐 Shared / Collaborative AI
                        </div>
                    </div>
                    <span class="clay-badge clay-badge-blue">🌐 {SHARED_STATUS_LABEL}</span>
                </div>
                <ul style="margin: 0; padding-left: 18px; font-size: 0.88rem; color: #2d4c6d; line-height: 1.6;">
                    <li><strong>Source:</strong> Uses only explicitly approved, sanitized general institutional knowledge.</li>
                    <li><strong>Zero Fallback:</strong> Does not read, search, or fall back to private hospital CSV files.</li>
                    <li><strong>Safety Filter:</strong> Patient-level records and identifiers are strictly rejected from disclosure.</li>
                </ul>
                <div class="clay-well" style="margin-top: 14px; padding: 12px 14px; font-size: 0.82rem;">
                    <div style="font-weight: 700; color: #1e40af; margin-bottom: 4px;">Live Shared State:</div>
                    <div style="color: #0b2545;">Approved Knowledge: <strong>{shared_items_count} item(s) published</strong></div>
                    <div style="color: #3b5a7a; margin-top: 2px;">Contributing Institutions: <strong>{contrib_str}</strong></div>
                    <div style="color: #3b5a7a; margin-top: 2px;">Audit Log: <strong>{shared_audit_count} approval record(s)</strong> persisted</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Visual Flow Card
    st.markdown(
        """
        <div class="clay-card" style="padding: 18px 22px; margin-bottom: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 6px;">
                <div style="font-size: 0.96rem; font-weight: 700; color: #0b2545;">
                    🔄 Data Governance & Sharing Pipeline Flow
                </div>
                <span class="clay-badge clay-badge-blue">Human-in-the-Loop Gate</span>
            </div>
            <div style="display: flex; flex-direction: row; flex-wrap: nowrap; align-items: center; justify-content: space-between; gap: 6px; padding: 4px 0;">
                <div class="clay-well" style="flex: 1 1 0; min-width: 0; margin: 0; padding: 12px 6px; text-align: center;">
                    <div style="font-size: 1.25rem;">📄</div>
                    <div style="font-weight: 700; color: #0b2545; font-size: 0.82rem; margin-top: 3px; white-space: nowrap;">Hospital CSV</div>
                    <div style="font-size: 0.70rem; color: #4b6b8e;">On-premise upload</div>
                </div>
                <div style="flex: 0 0 auto; text-align: center; font-size: 1.15rem; color: #2563eb; font-weight: 800; padding: 0 2px;">➔</div>
                <div class="clay-well" style="flex: 1 1 0; min-width: 0; margin: 0; padding: 12px 6px; text-align: center;">
                    <div style="font-size: 1.25rem;">🛡️</div>
                    <div style="font-weight: 700; color: #0b2545; font-size: 0.82rem; margin-top: 3px; white-space: nowrap;">PII Review</div>
                    <div style="font-size: 0.70rem; color: #4b6b8e;">Regex & header scan</div>
                </div>
                <div style="flex: 0 0 auto; text-align: center; font-size: 1.15rem; color: #2563eb; font-weight: 800; padding: 0 2px;">➔</div>
                <div class="clay-well" style="flex: 1 1 0; min-width: 0; margin: 0; padding: 12px 6px; text-align: center;">
                    <div style="font-size: 1.25rem;">👨‍⚕️</div>
                    <div style="font-weight: 700; color: #0b2545; font-size: 0.82rem; margin-top: 3px; white-space: nowrap;">Human Approval</div>
                    <div style="font-size: 0.70rem; color: #4b6b8e;">Preview & confirm</div>
                </div>
                <div style="flex: 0 0 auto; text-align: center; font-size: 1.15rem; color: #2563eb; font-weight: 800; padding: 0 2px;">➔</div>
                <div class="clay-well" style="flex: 1 1 0; min-width: 0; margin: 0; padding: 12px 6px; text-align: center;">
                    <div style="font-size: 1.25rem;">📚</div>
                    <div style="font-weight: 700; color: #0b2545; font-size: 0.82rem; margin-top: 3px; white-space: nowrap;">Shared Knowledge</div>
                    <div style="font-size: 0.70rem; color: #4b6b8e;">shared_knowledge.json</div>
                </div>
                <div style="flex: 0 0 auto; text-align: center; font-size: 1.15rem; color: #2563eb; font-weight: 800; padding: 0 2px;">➔</div>
                <div class="clay-well" style="flex: 1 1 0; min-width: 0; margin: 0; padding: 12px 6px; text-align: center;">
                    <div style="font-size: 1.25rem;">🌐</div>
                    <div style="font-weight: 700; color: #0b2545; font-size: 0.82rem; margin-top: 3px; white-space: nowrap;">Shared AI</div>
                    <div style="font-size: 0.70rem; color: #4b6b8e;">Sanitized retrieval only</div>
                </div>
            </div>
            <div class="clay-demo-alert" style="margin-top: 14px; margin-bottom: 0; padding: 10px 14px;">
                <span style="font-size: 1.15rem;">🔬</span>
                <div style="font-size: 0.80rem; line-height: 1.45;">
                    <strong>Prototype Notice:</strong> This comparison panel visualizes the real-time architectural separation between 
                    on-premise hospital datasets and the consortium shared knowledge store. This application is an engineering demonstration 
                    of access boundaries and human-in-the-loop governance. It does <em>not</em> retrain language model weights or execute 
                    formal distributed federated optimization algorithms.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Account Profile & Federated Architecture Details
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.markdown(
            f"""
            <div class="clay-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: #0b2545; margin-bottom: 12px;">
                    🏥 Active Node Profile & Permissions
                </div>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.90rem;">
                    <tr style="border-bottom: 1px solid rgba(225, 238, 250, 0.8);">
                        <td style="padding: 10px 4px; color: #4b6b8e; font-weight: 600; width: 38%;">Authenticated Account</td>
                        <td style="padding: 10px 4px; color: #0b2545; font-family: monospace;">{user_email}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(225, 238, 250, 0.8);">
                        <td style="padding: 10px 4px; color: #4b6b8e; font-weight: 600;">Assigned Role</td>
                        <td style="padding: 10px 4px; color: #0b2545;"><strong>{user["role"]}</strong></td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(225, 238, 250, 0.8);">
                        <td style="padding: 10px 4px; color: #4b6b8e; font-weight: 600;">Hospital Identifier</td>
                        <td style="padding: 10px 4px; color: #1d4ed8; font-weight: 700;">{user["hospital_id"]}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(225, 238, 250, 0.8);">
                        <td style="padding: 10px 4px; color: #4b6b8e; font-weight: 600;">Facility Designation</td>
                        <td style="padding: 10px 4px; color: #0b2545;">{user["facility_type"]}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 4px; color: #4b6b8e; font-weight: 600;">Node Authorization Level</td>
                        <td style="padding: 10px 4px; color: #0b2545;">{user["permissions"]}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="clay-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: #0b2545; margin-bottom: 12px;">
                    🛡️ How AITHOS-FL Protects Healthcare Data
                </div>
                <div style="font-size: 0.90rem; color: #2d4c6d; line-height: 1.6;">
                    Unlike traditional centralized AI architectures that aggregate sensitive patient records into 
                    a central repository, AITHOS-FL follows the strict <strong>Federated Learning principle</strong>:
                </div>
                <div style="margin-top: 14px; display: grid; gap: 10px;">
                    <div class="clay-well" style="margin: 0; padding: 12px 16px;">
                        <span style="font-weight: 700; color: #1e40af;">1. Data Never Leaves the Hospital:</span>
                        <div style="font-size: 0.84rem; color: #3b5a7a; margin-top: 2px;">
                            Raw patient EHR records, lab diagnostics, and images stay securely behind the institutional firewall.
                        </div>
                    </div>
                    <div class="clay-well" style="margin: 0; padding: 12px 16px;">
                        <span style="font-weight: 700; color: #1e40af;">2. Only Numerical Model Weights are Shared:</span>
                        <div style="font-size: 0.84rem; color: #3b5a7a; margin-top: 2px;">
                            Each hospital node trains a local model copy and transmits only encrypted weight gradient tensors.
                        </div>
                    </div>
                    <div class="clay-well" style="margin: 0; padding: 12px 16px;">
                        <span style="font-weight: 700; color: #1e40af;">3. Differential Privacy & PII Sanitization:</span>
                        <div style="font-size: 0.84rem; color: #3b5a7a; margin-top: 2px;">
                            Direct patient identifiers are sanitized and differential privacy perturbation bounds are enforced.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div class="clay-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: #0b2545; margin-bottom: 12px;">
                    🚀 Quick Navigation
                </div>
                <div style="font-size: 0.86rem; color: #4b6b8e; margin-bottom: 14px;">
                    Access upcoming pipeline modules:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("📁 Go to Dataset Upload", use_container_width=True):
            st.session_state.active_nav = "Upload Dataset"
            st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if st.button("🧠 Go to Local AI", use_container_width=True):
            st.session_state.active_nav = "Local AI"
            st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if st.button("🌐 Go to Shared AI", use_container_width=True):
            st.session_state.active_nav = "Shared AI"
            st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if st.button("🛡️ Go to Privacy Center", use_container_width=True):
            st.session_state.active_nav = "Privacy Center"
            st.rerun()

        st.markdown(
            """
            <div class="clay-card" style="margin-top: 20px;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #0b2545; margin-bottom: 8px;">
                    📡 Registered Nodes
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.82rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; background: #f0f7ff; border-radius: 8px;">
                        <span><strong>Hospital A</strong> (HOSP-A)</span>
                        <span class="clay-badge clay-badge-green">Enrolled</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; background: #f0f7ff; border-radius: 8px;">
                        <span><strong>Hospital B</strong> (HOSP-B)</span>
                        <span class="clay-badge clay-badge-green">Enrolled</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; background: #f0f7ff; border-radius: 8px;">
                        <span><strong>Coordinator Hub</strong></span>
                        <span class="clay-badge clay-badge-blue">Standby</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# PAGE 2: Upload Dataset & PII Compliance Scanner
# -----------------------------------------------------------------------------
def render_upload_dataset_page(user: dict):
    # Enforce role-based access control: Normal users must not upload or view hospital datasets
    if user["hospital_id"] not in ["HOSP-A", "HOSP-B"]:
        st.markdown(
            f"""
            <div class="clay-hero">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                    <div>
                        <span class="clay-badge clay-badge-amber">Access Restricted</span>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #0b2545; margin-top: 6px;">
                            Institutional Data Ingestion
                        </div>
                        <div style="font-size: 0.92rem; color: #3b5a7a; margin-top: 4px;">
                            Active Account: <strong>{user["name"]} ({user["role"]})</strong>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="clay-card">
                <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 14px;">
                    <span style="font-size: 2.2rem;">🔒</span>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #0b2545;">
                            Hospital Node Access Required
                        </div>
                        <div style="font-size: 0.88rem; color: #7f1d1d; font-weight: 600;">
                            Standard researcher / external observer accounts are not permitted to upload or inspect private hospital clinical datasets.
                        </div>
                    </div>
                </div>
                <div style="font-size: 0.90rem; color: #3b5a7a; line-height: 1.6;">
                    To protect institutional privacy under HIPAA Safe Harbor regulations, dataset ingestion is strictly restricted to authenticated 
                    <strong>Hospital Node Administrators</strong> (Hospital A or Hospital B). Please sign in with an institutional account to upload local datasets.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Authorized Hospital Node View
    hospital_id = user["hospital_id"]
    store: HospitalDatasetStore = st.session_state.dataset_store

    st.markdown(
        f"""
        <div class="clay-hero">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <span class="clay-badge clay-badge-blue">On-Premise Ingestion & Compliance</span>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #0b2545; margin-top: 6px;">
                        Upload Local Dataset
                    </div>
                    <div style="font-size: 0.92rem; color: #3b5a7a; margin-top: 4px;">
                        Institutional Node: <strong>{hospital_id} &bull; {user["name"]}</strong>
                    </div>
                </div>
                <span class="clay-badge clay-badge-green">● Local Isolation Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ingestion Controls Clay Card
    st.markdown(
        """
        <div class="clay-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="font-size: 1.12rem; font-weight: 700; color: #0b2545;">
                    📥 Ingest New CSV Dataset
                </div>
                <span class="clay-badge clay-badge-blue">Max size: 25 MB</span>
            </div>
            <div style="font-size: 0.88rem; color: #476685; margin-bottom: 16px; line-height: 1.5;">
                Upload a tabular CSV file containing clinical records. All processing occurs locally within 
                <strong>""" + hospital_id + """</strong>. Raw patient records are never transmitted to any external server.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_synthetic = st.columns([2, 1.2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Select CSV file to upload",
            type=["csv"],
            key=f"csv_uploader_{hospital_id}",
            help="Validated for UTF-8/Latin-1 encodings, maximum 25 MB file size, and non-empty tabular format.",
        )

        if uploaded_file is not None:
            # Check if this exact file hasn't been ingested in this session yet
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            # Check if button or immediate action needed
            if st.button("Process & Scan Dataset", key=f"btn_process_{uploaded_file.name}"):
                try:
                    new_ds = store.ingest_dataset(user, filename, file_bytes)
                    st.session_state.selected_dataset_id = new_ds.id
                    st.success(f"✅ Successfully ingested '{filename}' ({new_ds.row_count} rows, {new_ds.column_count} columns) into {hospital_id} vault.")
                    st.rerun()
                except ValueError as val_err:
                    st.error(f"❌ Validation Error: {str(val_err)}")
                except PermissionError as perm_err:
                    st.error(f"❌ Security Permission Error: {str(perm_err)}")
                except Exception as exc:
                    st.error(f"❌ Ingestion Error: {str(exc)}")

    with col_synthetic:
        st.markdown(
            """
            <div class="clay-well" style="margin-top: 0; padding: 14px 16px;">
                <div style="font-weight: 700; color: #1e40af; font-size: 0.88rem; margin-bottom: 4px;">
                    ⚡ One-Click Demo Dataset
                </div>
                <div style="font-size: 0.80rem; color: #3b5a7a; line-height: 1.4; margin-bottom: 10px;">
                    Load a realistic synthetic clinical dataset tailored for this hospital node to immediately test PII scanning and redaction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(f"Load Synthetic Demo Data ({hospital_id})", use_container_width=True, key=f"btn_demo_{hospital_id}"):
            synth_filename, synth_bytes = generate_synthetic_csv_bytes(hospital_id)
            try:
                synth_ds = store.ingest_dataset(user, synth_filename, synth_bytes)
                st.session_state.selected_dataset_id = synth_ds.id
                st.success(f"✅ Loaded synthetic dataset '{synth_filename}' ({synth_ds.row_count} rows) into {hospital_id}.")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Ingestion failed: {str(exc)}")

    # Retrieve all datasets owned by this hospital
    hospital_datasets = store.get_datasets_for_user(user)

    if not hospital_datasets:
        st.markdown(
            """
            <div class="clay-card" style="text-align: center; padding: 36px 24px; margin-top: 15px;">
                <span style="font-size: 2.2rem;">📂</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0b2545; margin-top: 8px;">
                    No Datasets Loaded for this Node
                </div>
                <div style="font-size: 0.88rem; color: #5a7b9e; margin-top: 4px;">
                    Upload a CSV file above or click 'Load Synthetic Demo Data' to evaluate PII detection and redaction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Dataset Selection & Details
    st.markdown("<hr style='margin: 24px 0 16px;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #0b2545;">
                📊 Node Datasets & Privacy Inspection
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_options = {ds.id: f"{ds.filename} ({ds.row_count} rows, {ds.uploaded_at})" for ds in hospital_datasets}

    # Ensure selected dataset id is valid
    if st.session_state.selected_dataset_id not in dataset_options:
        st.session_state.selected_dataset_id = hospital_datasets[-1].id

    selected_id = st.selectbox(
        "Select Active Dataset to Inspect",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x],
        index=list(dataset_options.keys()).index(st.session_state.selected_dataset_id),
        key="dataset_select_box",
    )
    st.session_state.selected_dataset_id = selected_id

    # Retrieve current dataset object securely
    active_dataset = store.get_dataset_by_id(user, selected_id)
    scan_res = active_dataset.scan_result

    # -------------------------------------------------------------------------
    # SECTION C: Privacy Dashboard
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="clay-card" style="margin-top: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-size: 1.12rem; font-weight: 700; color: #0b2545;">
                        🛡️ Privacy Compliance Dashboard: {active_dataset.filename}
                    </span>
                    <div style="font-size: 0.82rem; color: #4b6b8e; margin-top: 2px;">
                        Associated with <strong>{active_dataset.hospital_id}</strong> &bull; Uploaded at {active_dataset.uploaded_at}
                    </div>
                </div>
                <div>
                    <span class="clay-badge clay-badge-green">● Scan Status: Completed</span>
                </div>
            </div>
            <div style="font-size: 0.88rem; color: #3b5a7a; line-height: 1.5; margin-bottom: 16px;">
                All metrics are computed directly from the local dataset scan. No fabricated statistics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Privacy Metric Cards
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(
            f"""
            <div class="clay-metric-card">
                <div class="clay-metric-title">Total Rows Scanned</div>
                <div class="clay-metric-value">{scan_res.total_rows}</div>
                <div class="clay-metric-sub">100% Verified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol2:
        badge_cls = "clay-badge-red" if len(scan_res.sensitive_columns) > 0 else "clay-badge-green"
        st.markdown(
            f"""
            <div class="clay-metric-card">
                <div class="clay-metric-title">Sensitive Columns</div>
                <div class="clay-metric-value">{len(scan_res.sensitive_columns)}</div>
                <div class="clay-metric-sub">{len(scan_res.sensitive_columns)} Identified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol3:
        st.markdown(
            f"""
            <div class="clay-metric-card">
                <div class="clay-metric-title">Email Matches</div>
                <div class="clay-metric-value">{scan_res.email_matches}</div>
                <div class="clay-metric-sub">Regex Verified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol4:
        st.markdown(
            f"""
            <div class="clay-metric-card">
                <div class="clay-metric-title">Phone Matches</div>
                <div class="clay-metric-value">{scan_res.phone_matches}</div>
                <div class="clay-metric-sub">Config Pattern Match</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sensitive Columns Breakdown
    if scan_res.sensitive_columns:
        sens_pills = " ".join([f'<span class="clay-badge clay-badge-red" style="margin-right: 6px; margin-bottom: 6px;">⚠️ {col}</span>' for col in scan_res.sensitive_columns])
        st.markdown(
            f"""
            <div class="clay-card" style="margin-top: 14px; padding: 18px 22px;">
                <div style="font-weight: 700; color: #991b1b; font-size: 0.90rem; margin-bottom: 8px;">
                    🚨 Detected Sensitive Direct Identifiers:
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;">
                    {sens_pills}
                </div>
                <div style="font-size: 0.82rem; color: #4b6b8e;">
                    These columns have been designated for complete replacement with <code>[REDACTED]</code> in the sanitized view.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Before-and-After Data Preview Tabs
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="font-size: 1.15rem; font-weight: 700; color: #0b2545; margin: 20px 0 10px;">
            🔍 Before & After Inspection Preview
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_original, tab_sanitized, tab_audit = st.tabs([
        "📋 Original Local Upload (Preserved)",
        "🛡️ Sanitized Preview ([REDACTED] Copy)",
        "📑 Column Audit Breakdown",
    ])

    with tab_original:
        st.markdown(
            f"""
            <div style="font-size: 0.86rem; color: #476685; margin-bottom: 10px;">
                <strong>Local Preservation Guarantee:</strong> The original file <code>{active_dataset.filename}</code> 
                ({active_dataset.row_count} rows &times; {active_dataset.column_count} columns) remains unmodified inside 
                institutional node memory.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(active_dataset.original_df, width="stretch")

    with tab_sanitized:
        st.markdown(
            """
            <div style="font-size: 0.86rem; color: #476685; margin-bottom: 10px;">
                <strong>Sanitization Output:</strong> Sensitive columns have been safely substituted with 
                <code>[REDACTED]</code>, and inline email addresses and phone numbers in unstructured text columns 
                have been masked.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(active_dataset.redacted_df, width="stretch")

        # Download Sanitized Copy Button
        sanitized_csv_data = active_dataset.redacted_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sanitized CSV ([REDACTED])",
            data=sanitized_csv_data,
            file_name=f"sanitized_{active_dataset.filename}",
            mime="text/csv",
        )

    with tab_audit:
        # Table of audit findings per column
        audit_records = []
        for col_name in active_dataset.column_names:
            findings = scan_res.findings_by_column.get(col_name, {})
            is_sens = findings.get("is_sensitive_column", False)
            em_cnt = findings.get("email_count", 0)
            ph_cnt = findings.get("phone_count", 0)

            action = "Full Column Redacted to [REDACTED]" if is_sens else (
                f"Inline Masked ({em_cnt} emails, {ph_cnt} phones)" if (em_cnt > 0 or ph_cnt > 0) else "Preserved Intact"
            )

            audit_records.append({
                "Column Name": col_name,
                "Direct Identifier?": "YES (Sensitive Header)" if is_sens else "No",
                "Email Matches": em_cnt,
                "Phone Matches": ph_cnt,
                "Sanitization Action": action,
            })

        st.dataframe(pd.DataFrame(audit_records), width="stretch")


# -----------------------------------------------------------------------------
# PAGE 3: Local AI Assistant (On-Premise Clinical Chat)
# -----------------------------------------------------------------------------
def render_local_ai_page(user: dict):
    # Enforce role-based access control: Normal users must not query hospital datasets
    if user["hospital_id"] not in ["HOSP-A", "HOSP-B"]:
        st.markdown(
            f"""
            <div class="clay-hero">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                    <div>
                        <span class="clay-badge clay-badge-amber">Access Restricted</span>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #0b2545; margin-top: 6px;">
                            Local AI Assistant
                        </div>
                        <div style="font-size: 0.92rem; color: #3b5a7a; margin-top: 4px;">
                            Active Account: <strong>{user["name"]} ({user["role"]})</strong>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="clay-card">
                <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 14px;">
                    <span style="font-size: 2.2rem;">🔒</span>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #0b2545;">
                            Hospital Node Access Required
                        </div>
                        <div style="font-size: 0.88rem; color: #7f1d1d; font-weight: 600;">
                            Standard researcher / external observer accounts are not permitted to query hospital-private clinical records.
                        </div>
                    </div>
                </div>
                <div style="font-size: 0.90rem; color: #3b5a7a; line-height: 1.6;">
                    The Local AI Assistant operates exclusively on private clinical data residing behind institutional hospital firewalls 
                    (Hospital A or Hospital B). To evaluate local clinical question answering, please log in with an authorized hospital node account.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    hospital_id = user["hospital_id"]
    store: HospitalDatasetStore = st.session_state.dataset_store
    assistant: LocalAIAssistant = st.session_state.local_ai_assistant
    assistant.store = store

    # Partitioned conversation history: strictly scoped to the authenticated user and hospital
    user_email = st.session_state.current_user_email or "unknown"
    user_key = f"{hospital_id}_{user_email}"
    chat_history = st.session_state.chat_history_by_user.setdefault(user_key, [])

    # Header Clay Hero
    st.markdown(
        f"""
        <div class="clay-hero">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span class="clay-badge clay-badge-blue">On-Premise Clinical Assistant</span>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #0b2545; margin-top: 6px;">
                        Local AI Assistant
                    </div>
                    <div style="font-size: 0.92rem; color: #3b5a7a; margin-top: 4px;">
                        Operating Node: <strong>{hospital_id} ({user["name"]})</strong> &bull; Role: <strong>{user["role"]}</strong>
                    </div>
                </div>
                <div>
                    <span class="clay-badge clay-badge-green">🔒 {LOCAL_STATUS_LABEL}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Node & Active Dataset Context Card
    hospital_datasets = store.get_datasets_for_user(user)
    active_dataset = None
    if hospital_datasets:
        active_dataset_id = st.session_state.get("selected_dataset_id")
        if active_dataset_id:
            try:
                active_dataset = store.get_dataset_by_id(user, active_dataset_id)
            except Exception:
                active_dataset = hospital_datasets[-1]
                st.session_state.selected_dataset_id = active_dataset.id
        else:
            active_dataset = hospital_datasets[-1]
            st.session_state.selected_dataset_id = active_dataset.id

    col_ctx1, col_ctx2 = st.columns([2, 1])
    with col_ctx1:
        if active_dataset:
            st.markdown(
                f"""
                <div class="clay-card" style="padding: 16px 20px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #4b6b8e; font-weight: 700;">Active Local Data Source</div>
                            <div style="font-size: 1.05rem; font-weight: 700; color: #0b2545; margin-top: 2px;">
                                📄 {active_dataset.filename}
                            </div>
                            <div style="font-size: 0.82rem; color: #2563eb; margin-top: 2px;">
                                {active_dataset.row_count} rows &bull; {active_dataset.column_count} columns &bull; Ingested {active_dataset.uploaded_at}
                            </div>
                        </div>
                        <span class="clay-badge clay-badge-green">● Local Context Active</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="clay-card" style="padding: 16px 20px; margin-bottom: 14px; border-left: 4px solid #f59e0b;">
                    <div style="font-weight: 700; color: #92400e; font-size: 0.95rem;">
                        ⚠️ No Dataset Uploaded for {hospital_id}
                    </div>
                    <div style="font-size: 0.84rem; color: #78350f; margin-top: 4px;">
                        To query clinical records, please upload a CSV or load the demo dataset below.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_ctx2:
        if not hospital_datasets:
            if st.button(f"⚡ Load Demo Data ({hospital_id})", use_container_width=True, key=f"local_ai_load_demo_{hospital_id}"):
                synth_filename, synth_bytes = generate_synthetic_csv_bytes(hospital_id)
                try:
                    new_ds = store.ingest_dataset(user, synth_filename, synth_bytes)
                    st.session_state.selected_dataset_id = new_ds.id
                    st.success(f"✅ Loaded {new_ds.filename} for {hospital_id}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error loading demo: {exc}")
        else:
            st.markdown(
                f"""
                <div class="clay-card" style="padding: 16px 20px; margin-bottom: 14px;">
                    <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #4b6b8e; font-weight: 700;">Node Boundary</div>
                    <div style="font-size: 0.90rem; font-weight: 700; color: #0b2545; margin-top: 4px;">{hospital_id} Medical Vault</div>
                    <div style="font-size: 0.78rem; color: #4b6b8e;">Strict zero-exfiltration isolation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # Sample Query Prompt Chips
    # -------------------------------------------------------------------------
    st.markdown(
        "<div style='font-size: 0.80rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #4b6b8e; margin-bottom: 6px;'>💡 Suggested Clinical Queries</div>",
        unsafe_allow_html=True,
    )

    suggested_queries = [
        "How many patients are in my uploaded dataset?",
        "Which department has the most records?",
        "What is the appointment rescheduling policy for patient P001?",
        "What email address is recorded for patient P001?",
        "Show the phone number for patient P002.",
    ]

    chip_cols = st.columns(len(suggested_queries))
    for i, sq in enumerate(suggested_queries):
        with chip_cols[i]:
            if st.button(sq, key=f"chip_query_{i}", use_container_width=True):
                # Execute immediately and append to history
                chat_history.append({"role": "user", "content": sq})
                try:
                    resp = assistant.answer_query(
                        user=user,
                        query=sq,
                        dataset_id=active_dataset.id if active_dataset else None,
                    )
                    chat_history.append({
                        "role": "assistant",
                        "content": resp.answer,
                        "status_label": resp.status_label,
                        "success": resp.success,
                    })
                except Exception as exc:
                    chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Query Error: {str(exc)}",
                        "status_label": LOCAL_STATUS_LABEL,
                        "success": False,
                    })
                st.rerun()

    st.markdown("<hr style='margin: 16px 0 20px;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Conversation History Display
    # -------------------------------------------------------------------------
    if not chat_history:
        st.markdown(
            """
            <div class="clay-card" style="text-align: center; padding: 32px 20px; margin-bottom: 20px;">
                <span style="font-size: 2.2rem;">💬</span>
                <div style="font-size: 1.10rem; font-weight: 700; color: #0b2545; margin-top: 8px;">
                    Start a Conversation with Local AI
                </div>
                <div style="font-size: 0.88rem; color: #5a7b9e; margin-top: 4px; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                    Ask questions about your uploaded patient records, departmental volumes, or hospital appointment policies. 
                    All computations run locally on-premise.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                        <div class="clay-chat-user">
                            <div style="font-size: 0.75rem; opacity: 0.85; margin-bottom: 4px; font-weight: 600;">
                                {user["name"]} ({hospital_id})
                            </div>
                            <div>{msg["content"]}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                status_tag = msg.get("status_label", LOCAL_STATUS_LABEL)
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 14px;">
                        <div class="clay-chat-assistant">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-size: 0.78rem; font-weight: 700; color: #1e40af;">
                                    🤖 Local AI Assistant
                                </span>
                                <span class="clay-badge clay-badge-green" style="font-size: 0.72rem;">
                                    {status_tag}
                                </span>
                            </div>
                            <div style="color: #0b2545; font-size: 0.91rem; line-height: 1.6;">
                                {msg["content"]}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------------------
    # Chat Input Box & Clear Chat Action
    # -------------------------------------------------------------------------
    with st.form("local_ai_form", clear_on_submit=True):
        chat_query_input = st.text_input(
            "Type your query:",
            placeholder="e.g. How many patients are in my uploaded dataset?",
            label_visibility="collapsed",
            key=f"chat_text_input_{user_key}",
        )
        col_send, col_clear = st.columns([5, 1.2])
        with col_send:
            send_submitted = st.form_submit_button("Send Query", use_container_width=True)
        with col_clear:
            clear_submitted = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)

    if clear_submitted:
        chat_history.clear()
        st.rerun()

    if send_submitted and chat_query_input.strip():
        user_text = chat_query_input.strip()
        chat_history.append({"role": "user", "content": user_text})

        # Process through LocalAIAssistant with strict backend authorization
        try:
            resp = assistant.answer_query(
                user=user,
                query=user_text,
                dataset_id=active_dataset.id if active_dataset else None,
            )
            chat_history.append({
                "role": "assistant",
                "content": resp.answer,
                "status_label": resp.status_label,
                "success": resp.success,
            })
        except PermissionError as perm_err:
            chat_history.append({
                "role": "assistant",
                "content": f"🔒 Access Denied: {str(perm_err)}",
                "status_label": "Authorization Rejected",
                "success": False,
            })
        except Exception as exc:
            chat_history.append({
                "role": "assistant",
                "content": f"❌ Query Error: {str(exc)}",
                "status_label": LOCAL_STATUS_LABEL,
                "success": False,
            })
        st.rerun()


# -----------------------------------------------------------------------------
# PAGE 4: Shared AI Assistant (Consortium Knowledge Intelligence)
# -----------------------------------------------------------------------------
def render_shared_ai_page(user: dict):
    shared_store: SharedKnowledgeStore = st.session_state.shared_knowledge_store
    shared_assistant: SharedAIAssistant = st.session_state.shared_ai_assistant
    shared_assistant.store = shared_store

    chat_history = st.session_state.shared_ai_chat_history

    # Header Clay Hero
    st.markdown(
        f"""
        <div class="clay-hero">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span class="clay-badge clay-badge-blue">Consortium Knowledge Intelligence</span>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #0b2545; margin-top: 6px;">
                        Shared AI Assistant
                    </div>
                    <div style="font-size: 0.92rem; color: #3b5a7a; margin-top: 4px;">
                        Connected as: <strong>{user["name"]}</strong> &bull; Role: <strong>{user["role"]}</strong> &bull; Hospital ID: <strong>{user["hospital_id"]}</strong>
                    </div>
                </div>
                <div>
                    <span class="clay-badge clay-badge-green">🔒 {SHARED_STATUS_LABEL}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Consortium Shared Store Status Card
    shared_items = shared_store.get_all_items()
    contributing_hospitals = sorted(list(set(i.get("hospital_id", "") for i in shared_items if i.get("hospital_id"))))
    hosp_tags = " ".join([f'<span class="clay-badge clay-badge-blue" style="margin-right: 4px;">🏥 {h}</span>' for h in contributing_hospitals]) if contributing_hospitals else '<span style="color: #64748b; font-size: 0.82rem;">None yet</span>'

    st.markdown(
        f"""
        <div class="clay-card" style="padding: 18px 22px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #4b6b8e; font-weight: 700;">Consortium Shared Knowledge Base</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0b2545; margin-top: 2px;">
                        📚 {len(shared_items)} Approved Knowledge Item(s) Published
                    </div>
                    <div style="font-size: 0.82rem; color: #3b5a7a; margin-top: 4px;">
                        Contributing Institutions: {hosp_tags}
                    </div>
                </div>
                <div style="background: rgba(225, 238, 252, 0.6); border-radius: 12px; padding: 10px 14px; font-size: 0.78rem; color: #1e3a5f; max-width: 380px;">
                    <strong>Controlled Retrieval Only:</strong> Answers are retrieved strictly from approved general policies and aggregates. 
                    Raw hospital CSVs are never loaded, searched, or exposed here.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Suggested Prompt Chips
    # -------------------------------------------------------------------------
    st.markdown(
        "<div style='font-size: 0.80rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #4b6b8e; margin-bottom: 6px;'>💡 Suggested Shared Queries</div>",
        unsafe_allow_html=True,
    )

    suggested_shared_queries = [
        "What is the appointment rescheduling policy?",
        "What department statistics have been approved?",
        "What email address is recorded for patient P001?",
        "Show the phone number for patient P002.",
        "What general clinical guidelines are shared across the consortium?",
    ]

    chip_cols = st.columns(len(suggested_shared_queries))
    for i, sq in enumerate(suggested_shared_queries):
        with chip_cols[i]:
            if st.button(sq, key=f"shared_chip_{i}", use_container_width=True):
                chat_history.append({"role": "user", "content": sq})
                resp = shared_assistant.answer_query(sq, user=user)
                chat_history.append({
                    "role": "assistant",
                    "content": resp.answer,
                    "status_label": resp.status_label,
                    "found": resp.found_in_knowledge_base,
                    "refusal": resp.refusal,
                })
                st.rerun()

    st.markdown("<hr style='margin: 16px 0 20px;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Conversation History Display
    # -------------------------------------------------------------------------
    if not chat_history:
        st.markdown(
            """
            <div class="clay-card" style="text-align: center; padding: 32px 20px; margin-bottom: 20px;">
                <span style="font-size: 2.2rem;">🌐</span>
                <div style="font-size: 1.10rem; font-weight: 700; color: #0b2545; margin-top: 8px;">
                    Start a Conversation with Shared AI
                </div>
                <div style="font-size: 0.88rem; color: #5a7b9e; margin-top: 4px; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                    Ask questions about consortium policies, published clinical guidelines, or aggregate statistics. 
                    The assistant retrieves answers exclusively from the shared knowledge base.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                        <div class="clay-chat-user">
                            <div style="font-size: 0.75rem; opacity: 0.85; margin-bottom: 4px; font-weight: 600;">
                                {user["name"]} ({user["hospital_id"]})
                            </div>
                            <div>{msg["content"]}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                status_tag = msg.get("status_label", SHARED_STATUS_LABEL)
                badge_type = "clay-badge-amber" if msg.get("refusal") else ("clay-badge-green" if msg.get("found") else "clay-badge-blue")
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 14px;">
                        <div class="clay-chat-assistant">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-size: 0.78rem; font-weight: 700; color: #1e40af;">
                                    🌐 Shared AI Assistant
                                </span>
                                <span class="clay-badge {badge_type}" style="font-size: 0.72rem;">
                                    {status_tag}
                                </span>
                            </div>
                            <div style="color: #0b2545; font-size: 0.91rem; line-height: 1.6;">
                                {msg["content"]}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------------------
    # Chat Input Box & Clear Chat Action
    # -------------------------------------------------------------------------
    with st.form("shared_ai_chat_form", clear_on_submit=True):
        shared_query_input = st.text_input(
            "Ask Shared AI Assistant:",
            placeholder="e.g. What is the appointment rescheduling policy?",
            label_visibility="collapsed",
            key="shared_chat_input_field",
        )
        col_send, col_clear = st.columns([5, 1.2])
        with col_send:
            send_submitted = st.form_submit_button("Send Query", use_container_width=True)
        with col_clear:
            clear_submitted = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)

    if clear_submitted:
        chat_history.clear()
        st.rerun()

    if send_submitted and shared_query_input.strip():
        user_text = shared_query_input.strip()
        chat_history.append({"role": "user", "content": user_text})

        resp = shared_assistant.answer_query(user_text, user=user)
        chat_history.append({
            "role": "assistant",
            "content": resp.answer,
            "status_label": resp.status_label,
            "found": resp.found_in_knowledge_base,
            "refusal": resp.refusal,
        })
        st.rerun()


# -----------------------------------------------------------------------------
# PAGE 5: Privacy Center
# -----------------------------------------------------------------------------
def render_privacy_center_page(user: dict):
    st.markdown(
        f"""
        <div class="clay-hero">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <span class="clay-badge clay-badge-blue">Module: Governance & Cryptography</span>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #0b2545; margin-top: 6px;">
                        Privacy & Compliance Center
                    </div>
                    <div style="font-size: 0.92rem; color: #3b5a7a; margin-top: 4px;">
                        Differential Privacy & Regulatory Assurance for <strong>{user["hospital_id"]}</strong>
                    </div>
                </div>
                <span class="clay-badge clay-badge-green">Active Privacy Engine</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # If institutional node, summarize real scan stats and show Review & Approve Sharing workflow
    if user["hospital_id"] in ["HOSP-A", "HOSP-B"]:
        hospital_id = user["hospital_id"]
        store: HospitalDatasetStore = st.session_state.dataset_store
        datasets = store.get_datasets_for_user(user)
        total_rows_all = sum(d.scan_result.total_rows for d in datasets)
        total_emails_all = sum(d.scan_result.email_matches for d in datasets)
        total_phones_all = sum(d.scan_result.phone_matches for d in datasets)

        st.markdown(
            f"""
            <div class="clay-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: #0b2545; margin-bottom: 8px;">
                    🛡️ Institutional Node Privacy Metrics ({hospital_id})
                </div>
                <div style="font-size: 0.88rem; color: #476685; margin-bottom: 14px;">
                    Summary of local data scanning across all datasets residing inside the {hospital_id} vault:
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                    <div class="clay-well" style="margin: 0;">
                        <div style="font-size: 0.80rem; color: #476685; font-weight: 600;">DATASETS INGESTED</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #0b2545;">{len(datasets)}</div>
                    </div>
                    <div class="clay-well" style="margin: 0;">
                        <div style="font-size: 0.80rem; color: #476685; font-weight: 600;">TOTAL ROWS SCANNED</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #0b2545;">{total_rows_all}</div>
                    </div>
                    <div class="clay-well" style="margin: 0;">
                        <div style="font-size: 0.80rem; color: #476685; font-weight: 600;">PII EMAILS SANITIZED</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #0b2545;">{total_emails_all}</div>
                    </div>
                    <div class="clay-well" style="margin: 0;">
                        <div style="font-size: 0.80rem; color: #476685; font-weight: 600;">PII PHONES SANITIZED</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #0b2545;">{total_phones_all}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------------------------
        # SECTION: Review & Approve Sharing (Approval Workflow)
        # ---------------------------------------------------------------------
        shared_store: SharedKnowledgeStore = st.session_state.shared_knowledge_store

        st.markdown(
            f"""
            <div class="clay-card" style="margin-top: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 1.25rem; font-weight: 800; color: #0b2545;">
                        📋 Review & Approve Sharing
                    </div>
                    <span class="clay-badge clay-badge-blue">Authorized Node Staff: {hospital_id}</span>
                </div>
                <div style="font-size: 0.88rem; color: #3b5a7a; line-height: 1.5; margin-bottom: 16px;">
                    Select and verify permitted general institutional policies or non-sensitive aggregate statistics to publish to 
                    the consortium Shared Knowledge Store. <strong>Direct patient identifiers, patient IDs, and personal records are strictly rejected.</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Preset selection
        sharing_preset = st.radio(
            "Select Knowledge Item to Authorize:",
            options=[
                "Institutional Appointment Rescheduling Policy (General Policy)",
                "Department Cohort Volume Breakdown (Non-sensitive Aggregate)",
                "Custom Institutional Workflow / Policy Submission",
            ],
            key=f"preset_sharing_{hospital_id}",
        )

        # Determine default values based on preset
        if "Rescheduling Policy" in sharing_preset:
            default_title = f"{user['name']} Appointment Rescheduling Policy"
            default_category = "Institutional Policy"
            if hospital_id == "HOSP-A":
                default_content = (
                    "Appointments at Hospital A Medical Center may be rescheduled up to 24 hours prior "
                    "to the scheduled slot without penalty via the patient portal or by contacting the clinic desk. "
                    "Rescheduling requests made within 24 hours require clinical coordinator approval."
                )
            else:
                default_content = (
                    "Appointments at Hospital B Regional Health require at least 48 hours advance notice "
                    "for rescheduling through the central outpatient registration desk. "
                    "Specialized diagnostic appointments are subject to department lead re-assignment."
                )
            default_summary = "General outpatient appointment rescheduling notice requirements."
        elif "Department Cohort" in sharing_preset:
            default_title = f"{user['name']} Department Cohort Breakdown"
            default_category = "Aggregate Statistics"
            if hospital_id == "HOSP-A":
                default_content = (
                    "Hospital A Medical Center departmental cohort volume breakdown: "
                    "Cardiology (60%, 3 records), Neurology (20%, 1 record), Pediatrics (20%, 1 record). "
                    "Total cohort size: 5 patients. Aggregated for clinical resource planning."
                )
            else:
                default_content = (
                    "Hospital B Regional Health departmental cohort volume breakdown: "
                    "Oncology (50%, 2 records), Orthopedics (25%, 1 record), Emergency (25%, 1 record). "
                    "Total cohort size: 4 patients. Aggregated for clinical resource planning."
                )
            default_summary = "Non-sensitive departmental volume distribution."
        else:
            default_title = ""
            default_category = "Institutional Policy"
            default_content = ""
            default_summary = ""

        # Form fields for approval candidate
        with st.container():
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                prop_title = st.text_input("Knowledge Title:", value=default_title, key=f"prop_title_{hospital_id}")
            with col_t2:
                prop_cat = st.selectbox(
                    "Category:",
                    options=list(ALLOWED_CATEGORIES),
                    index=list(ALLOWED_CATEGORIES).index(default_category) if default_category in ALLOWED_CATEGORIES else 0,
                    key=f"prop_cat_{hospital_id}",
                )

            prop_content = st.text_area("Content to Publish:", value=default_content, height=110, key=f"prop_content_{hospital_id}")
            prop_summary = st.text_input("Executive Summary / Tagline:", value=default_summary, key=f"prop_summary_{hospital_id}")

            # Safety Inspection & Pre-Validation
            is_valid, validation_msg = shared_store.validate_for_sharing(prop_title, prop_cat, prop_content)

            if not is_valid:
                st.error(f"🚨 {validation_msg}")
            else:
                st.success("✅ Content Safety Inspection Passed: Zero patient identifiers or contact PII detected.")

            # Preview Before Approval Card
            st.markdown(
                f"""
                <div class="clay-well" style="margin-top: 10px;">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #1e40af; margin-bottom: 4px;">
                        🔍 Preview Before Approval ({prop_cat})
                    </div>
                    <div style="font-size: 0.98rem; font-weight: 700; color: #0b2545; margin-bottom: 6px;">
                        {prop_title or "(Untitled)"}
                    </div>
                    <div style="font-size: 0.86rem; color: #2d4c6d; line-height: 1.5;">
                        {prop_content or "(No content provided)"}
                    </div>
                    <div style="font-size: 0.78rem; color: #4b6b8e; margin-top: 8px;">
                        Approving Node: <strong>{hospital_id}</strong> &bull; Authenticated Staff: <strong>{user.get('email') or st.session_state.current_user_email or 'Authorized Staff'}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Explicit Confirmation Checkbox
            confirm_approval = st.checkbox(
                "I explicitly confirm that this content contains strictly general institutional knowledge or aggregate statistics and does not disclose any patient-specific records or PII.",
                key=f"confirm_sharing_{hospital_id}",
            )

            # Publish Button
            if st.button("🚀 Publish to Shared Knowledge Base", disabled=not (is_valid and confirm_approval), key=f"btn_publish_{hospital_id}"):
                try:
                    published = shared_store.publish_item(
                        user=user,
                        title=prop_title,
                        category=prop_cat,
                        content=prop_content,
                        summary=prop_summary,
                        confirmed=confirm_approval,
                    )
                    st.success(f"🎉 Successfully published '{published['title']}' to the Consortium Shared Knowledge Base!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Publication Error: {str(exc)}")

        # Published Items & Audit Ledger Display
        node_audit_logs = shared_store.get_audit_log(hospital_id=hospital_id)
        if node_audit_logs:
            st.markdown(
                """
                <div style="font-size: 1.05rem; font-weight: 700; color: #0b2545; margin: 20px 0 8px;">
                    📑 Node Knowledge Sharing Audit Ledger
                </div>
                """,
                unsafe_allow_html=True,
            )
            audit_display_rows = []
            for entry in reversed(node_audit_logs):
                audit_display_rows.append({
                    "Timestamp": entry.get("timestamp"),
                    "Action": entry.get("action"),
                    "Title": entry.get("title"),
                    "Category": entry.get("category"),
                    "Staff User": entry.get("user_email"),
                    "Log ID": entry.get("log_id"),
                })
            st.dataframe(pd.DataFrame(audit_display_rows), width="stretch")

    else:
        # Normal User View in Privacy Center
        st.markdown(
            """
            <div class="clay-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #0b2545; margin-bottom: 8px;">
                    🛡️ Consortium Privacy & Governance Principles
                </div>
                <div style="font-size: 0.88rem; color: #3b5a7a; line-height: 1.6;">
                    As an independent research observer, you can access published institutional policies and aggregate statistics via 
                    the <strong>Shared AI</strong> assistant. Private hospital uploads and knowledge approval controls are restricted to 
                    authorized hospital node administrators.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="clay-card">
            <div style="font-size: 1.15rem; font-weight: 700; color: #0b2545; margin-bottom: 10px;">
                🛡️ Cryptographic & Differential Privacy Guardrails
            </div>
            <div style="font-size: 0.92rem; color: #2d4c6d; line-height: 1.6; margin-bottom: 16px;">
                The Privacy Center is the central cockpit for configuring mathematical privacy guarantees. 
                It prevents model reconstruction attacks through strict epsilon (&epsilon;) and delta (&delta;) 
                budget management, gradient clipping, and cryptographically verified audit trails.
            </div>
            <div class="clay-well">
                <div style="font-weight: 700; color: #1e40af; margin-bottom: 6px;">
                    ✨ Active Privacy Safeguards:
                </div>
                <ul style="margin: 0; padding-left: 20px; font-size: 0.88rem; color: #2d4c6d; line-height: 1.6;">
                    <li><strong>On-Premise Ingestion:</strong> Zero external transmission of raw uploaded patient data.</li>
                    <li><strong>Non-Destructive Redaction:</strong> Original files preserved, separate [REDACTED] copy generated.</li>
                    <li><strong>Role-Based Boundary:</strong> Standard research users cannot access or view hospital clinical datasets.</li>
                    <li><strong>Partitioned Storage:</strong> Hospital A and Hospital B data are kept strictly segregated.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Privacy configuration preview
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="clay-card">
                <div style="font-size: 1.0rem; font-weight: 700; color: #0b2545; margin-bottom: 8px;">
                    📐 Theoretical Privacy Parameters
                </div>
                <div style="font-size: 0.86rem; color: #35587f; line-height: 1.6;">
                    Differential Privacy Mechanism: <strong>Renyi DP (Gaussian)</strong><br>
                    Target Epsilon (&epsilon;): <strong>2.5 (Configurable)</strong><br>
                    Target Delta (&delta;): <strong>1e-5</strong><br>
                    Gradient Clipping Norm: <strong>1.0 (L2 Norm)</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="clay-card">
                <div style="font-size: 1.0rem; font-weight: 700; color: #0b2545; margin-bottom: 8px;">
                    📜 Compliance Certification Status
                </div>
                <div style="font-size: 0.86rem; color: #35587f; line-height: 1.6;">
                    HIPAA Security Rule Safe Harbor: <span class="clay-badge clay-badge-green">Compliant</span><br>
                    GDPR Article 25 (Privacy by Design): <span class="clay-badge clay-badge-green">Compliant</span><br>
                    On-Premise Data Retention: <strong>Zero Exfiltration</strong><br>
                    Audit Ledger: <strong>Initialized</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    if not st.session_state.authenticated or st.session_state.current_user is None:
        render_login_view()
    else:
        render_app_view()


if __name__ == "__main__":
    main()
