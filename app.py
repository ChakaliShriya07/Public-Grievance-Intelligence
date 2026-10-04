import os
import sys
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timezone
from textwrap import dedent

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "grievances_clean.csv"

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "public_grievance_intelligence"
MONGO_COLLECTION_NAME = "complaints"

sys.path.insert(0, str(BASE_DIR))

try:
    from src.live_intelligence import (
        get_all_live_complaints,
        get_live_summary,
        get_complaints_by_category,
        get_complaints_by_location,
        get_complaints_by_department,
        get_status_distribution,
        get_similar_complaint_stats,
        get_priority_complaints,
    )
except Exception:
    get_all_live_complaints = None
    get_live_summary = None
    get_complaints_by_category = None
    get_complaints_by_location = None
    get_complaints_by_department = None
    get_status_distribution = None
    get_similar_complaint_stats = None
    get_priority_complaints = None

try:
    from src.ai_grievance_engine import ask_ai_grievance_intelligence
except Exception:
    ask_ai_grievance_intelligence = None

st.set_page_config(
    page_title="Public Grievance Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: Inter, sans-serif; }
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(35,100,170,.18), transparent 28%),
                    radial-gradient(circle at 90% 80%, rgba(0,119,182,.10), transparent 30%),
                    linear-gradient(135deg,#06111f,#081827 48%,#06101c);
        color:#fff;
    }
    .block-container { max-width:1450px; padding-top:1rem; padding-bottom:3rem; }
    header[data-testid="stHeader"] { background:transparent; }
    footer,#MainMenu { visibility:hidden; }
    .gov-header,.card,.kpi,.insight {
        background:linear-gradient(145deg,rgba(14,39,61,.94),rgba(7,22,35,.94));
        border:1px solid rgba(110,180,225,.14);
        border-radius:16px;
        box-shadow:0 14px 40px rgba(0,0,0,.20);
    }
    .gov-header { padding:18px 22px; margin-bottom:22px; display:flex; justify-content:space-between; align-items:center; }
    .brand { display:flex; gap:13px; align-items:center; }
    .emblem { width:45px; height:45px; border-radius:11px; display:flex; align-items:center; justify-content:center; background:#103854; font-size:22px; }
    .brand-title { font-size:16px; font-weight:800; color:#eef8ff; }
    .brand-sub { font-size:9px; letter-spacing:1.4px; color:#7090a7; margin-top:3px; }
    .online { color:#70d693; font-size:10px; letter-spacing:1px; }
    .title { font-size:40px; font-weight:800; letter-spacing:-1.5px; color:#edf7ff; margin-top:15px; }
    .subtitle { color:#8ba4b8; font-size:14px; line-height:1.7; max-width:900px; margin-bottom:25px; }
    .section { margin:30px 0 13px; color:#82a0b7; font-size:10px; font-weight:800; letter-spacing:2px; }
    .kpi { padding:21px; min-height:130px; }
    .kpi-label { color:#6f8da4; font-size:10px; font-weight:800; letter-spacing:1.5px; }
    .kpi-value { color:#f2f9fd; font-size:30px; font-weight:800; margin-top:5px; }
    .kpi-note { color:#587288; font-size:10px; margin-top:5px; }
    .card { padding:22px; }
    .card-title { font-size:18px; font-weight:800; color:#e9f5fc; }
    .card-text { color:#819aae; font-size:12px; line-height:1.7; margin-top:7px; }
    .insight { padding:18px; margin-top:12px; }
    .badge { display:inline-block; padding:5px 9px; border-radius:999px; background:rgba(72,199,116,.10); color:#75d796; font-size:9px; font-weight:800; letter-spacing:1px; }
    .hero { padding:55px 20px 35px; text-align:center; }
    .hero-title { font-size:clamp(44px,6vw,78px); line-height:1.02; font-weight:800; letter-spacing:-3px; color:#eef8ff; }
    .hero-sub { max-width:760px; margin:20px auto; color:#92a9bc; line-height:1.8; }
    .tricolor { width:150px; height:3px; margin:25px auto; border-radius:5px; background:linear-gradient(90deg,#ff9933 0 33%,#fff 33% 66%,#138808 66%); }
    .small-muted { color:#718ca1; font-size:11px; }
    [data-testid="stMetric"] { background:rgba(8,25,40,.65); border:1px solid rgba(110,180,225,.10); padding:12px; border-radius:12px; }
    div.stButton > button { border-radius:9px; min-height:45px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
    return f"{salt}${digest}"


def verify_password(password, stored):
    try:
        salt, expected = stored.split("$", 1)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
        return actual == expected
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_historical_data():
    """
    Load only the historical fields required by the dashboard.

    The complete grievances_clean.csv contains large text columns.
    Loading all 13 columns at once can exhaust RAM on Windows and
    cause pandas' C parser to fail with an out-of-memory error.

    We therefore:
      1. read only the three fields used by this dashboard,
      2. read them in small chunks,
      3. concatenate only those small chunks.

    This preserves the historical totals, organization/state counts,
    resolution statistics and resolution-time chart without keeping
    the large grievance-description columns in memory.
    """
    if not DATA_PATH.exists():
        return pd.DataFrame(
            columns=["resolution_days", "org_code", "state"]
        )

    required_columns = [
        "resolution_days",
        "org_code",
        "state",
    ]

    try:
        chunks = []

        for chunk in pd.read_csv(
            DATA_PATH,
            usecols=required_columns,
            dtype={
                "org_code": "string",
                "state": "string",
            },
            chunksize=5000,
            engine="python",
            on_bad_lines="error",
        ):
            chunk["resolution_days"] = pd.to_numeric(
                chunk["resolution_days"],
                errors="coerce"
            )
            chunks.append(chunk)

        if not chunks:
            return pd.DataFrame(columns=required_columns)

        return pd.concat(chunks, ignore_index=True)

    except Exception as exc:
        raise RuntimeError(
            "Historical grievance data could not be loaded safely. "
            f"File: {DATA_PATH}. Details: {exc}"
        ) from exc


@st.cache_resource(show_spinner=False)
def get_mongo_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2500)
    client.admin.command("ping")
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    collection.create_index("grievance_id", unique=True)
    collection.create_index("citizen_email")
    collection.create_index("submitted_at")
    collection.create_index("status")
    return collection


def mongo_collection():
    try:
        return get_mongo_collection()
    except Exception:
        return None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def next_grievance_id(collection):
    year = datetime.now().year
    last = collection.find_one({"grievance_id": {"$regex": f"^PGI-{year}-"}}, sort=[("grievance_id", -1)])
    number = 1
    if last and last.get("grievance_id"):
        try:
            number = int(last["grievance_id"].split("-")[-1]) + 1
        except Exception:
            number = collection.count_documents({}) + 1
    return f"PGI-{year}-{number:06d}"


def init_state():
    defaults = {
        "page": "landing",
        "authenticated": False,
        "user_role": None,
        "user_email": None,
        "users": {
            "admin@grievance.gov": {"password": hash_password("Admin@123"), "role": "Admin", "name": "Administrator"}
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
historical = load_historical_data()


def historical_metrics():
    total = len(historical)
    resolved = int(historical["resolution_days"].notna().sum()) if "resolution_days" in historical.columns else 0
    avg = float(historical["resolution_days"].mean()) if resolved else 0
    median = float(historical["resolution_days"].median()) if resolved else 0
    orgs = int(historical["org_code"].nunique(dropna=True)) if "org_code" in historical.columns else 0
    states = int(historical["state"].nunique(dropna=True)) if "state" in historical.columns else 0
    return total, resolved, avg, median, orgs, states


def live_summary():
    if get_live_summary:
        try:
            return get_live_summary()
        except Exception:
            pass
    c = mongo_collection()
    if c is None:
        return {"total": 0, "submitted": 0, "under_review": 0, "resolved": 0, "rejected": 0, "unresolved": 0}
    docs = list(c.find({}, {"status": 1}))
    counts = pd.Series([d.get("status", "Submitted") for d in docs]).value_counts().to_dict() if docs else {}
    total = len(docs)
    resolved = int(counts.get("Resolved", 0))
    rejected = int(counts.get("Rejected", 0))
    return {
        "total": total,
        "submitted": int(counts.get("Submitted", 0)),
        "under_review": int(counts.get("Under Review", 0)),
        "resolved": resolved,
        "rejected": rejected,
        "unresolved": total - resolved - rejected,
    }


def get_citizen_complaints(email):
    c = mongo_collection()
    if c is None:
        return []
    return list(c.find({"citizen_email": email}).sort("submitted_at", -1))


def get_complaint(grievance_id):
    c = mongo_collection()
    if c is None:
        return None
    return c.find_one({"grievance_id": grievance_id.strip().upper()})


def similar_stats(doc):
    if get_similar_complaint_stats:
        try:
            return get_similar_complaint_stats(
                category=doc.get("category", ""),
                state=doc.get("state", ""),
                dist_name=doc.get("dist_name", ""),
            )
        except Exception:
            pass
    return {"count": 0, "resolved_count": 0, "resolved_percentage": 0, "median_resolution_days": None, "status_distribution": {}}


def format_resolution(value):
    if value is None or pd.isna(value):
        return "Not available"
    value = float(value)
    if value < 1:
        return "Less than 1 day"
    if value < 2:
        return "About 1 day"
    return f"{value:.1f} days"


def nav(items):
    cols = st.columns(len(items))
    for col, (label, page) in zip(cols, items):
        with col:
            if st.button(label, use_container_width=True, key=f"nav_{page}"):
                if page == "logout":
                    st.session_state.authenticated = False
                    st.session_state.user_role = None
                    st.session_state.user_email = None
                    st.session_state.page = "landing"
                else:
                    st.session_state.page = page
                st.rerun()


def header(title, subtitle, role=""):
    st.markdown(
        f"""
        <div class="gov-header">
            <div class="brand">
                <div class="emblem">🏛️</div>
                <div><div class="brand-title">{title}</div><div class="brand-sub">{subtitle}</div></div>
            </div>
            <div class="online">● {role.upper() or 'SYSTEM OPERATIONAL'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label, value, note):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)


def login_page():
    st.markdown('<div class="hero"><div class="badge">SECURE CIVIC INTELLIGENCE PLATFORM</div><div class="hero-title">Public Grievance<br>Intelligence</div><div class="tricolor"></div><div class="hero-sub">A citizen grievance portal and evidence-based administrative intelligence platform combining live grievance operations with historical government grievance patterns.</div></div>', unsafe_allow_html=True)
    total, resolved, avg, median, orgs, states = historical_metrics()
    a,b,c,d = st.columns(4)
    with a: render_kpi("HISTORICAL RECORDS", f"{total:,}", "Government grievance records")
    with b: render_kpi("ORGANIZATIONS", f"{orgs:,}", "Organizations represented")
    with c: render_kpi("REGIONS", f"{states:,}", "State/region coverage")
    with d: render_kpi("MEDIAN RESOLUTION", format_resolution(median), "Historical context")
    st.write("")
    if st.button("ENTER PLATFORM  →", type="primary", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()


def auth_page():
    header("Public Grievance Intelligence", "SECURE CITIZEN & ADMINISTRATOR ACCESS")
    t1, t2 = st.tabs(["🔐 Sign In", "📝 Create Citizen Account"])
    with t1:
        role = st.radio("Access role", ["Citizen", "Admin"], horizontal=True)
        email = st.text_input("Email address")
        password = st.text_input("Password", type="password")
        if st.button("SIGN IN  →", type="primary", use_container_width=True):
            account = st.session_state.users.get(email.strip().lower())
            if account and account["role"] == role and verify_password(password, account["password"]):
                st.session_state.authenticated = True
                st.session_state.user_role = role
                st.session_state.user_email = email.strip().lower()
                st.session_state.page = "citizen_portal" if role == "Citizen" else "admin_portal"
                st.rerun()
            else:
                st.error("Invalid email, password, or selected role.")
        st.info("Demo administrator: admin@grievance.gov / Admin@123")
    with t2:
        name = st.text_input("Full name")
        email = st.text_input("Email address", key="reg_email")
        password = st.text_input("Create password", type="password", key="reg_password")
        confirm = st.text_input("Confirm password", type="password")
        if st.button("CREATE CITIZEN ACCOUNT  →", use_container_width=True):
            email = email.strip().lower()
            if not name.strip() or not email or len(password) < 8:
                st.warning("Enter your name, email and a password of at least 8 characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif email in st.session_state.users:
                st.error("An account with this email already exists.")
            else:
                st.session_state.users[email] = {"password": hash_password(password), "role": "Citizen", "name": name.strip()}
                st.success("Citizen account created. You can now sign in.")
    if st.button("← Home"):
        st.session_state.page = "landing"
        st.rerun()


def citizen_nav():
    nav([
        ("🏠 Dashboard", "citizen_portal"),
        ("📝 Submit", "citizen_submit"),
        ("📋 My Complaints", "citizen_complaints"),
        ("📍 Track", "citizen_track"),
        ("🔔 Notifications", "citizen_notifications"),
        ("⭐ Feedback", "citizen_feedback"),
        ("🚪 Logout", "logout"),
    ])


def admin_nav():
    nav([
        ("🏠 Dashboard", "admin_portal"),
        ("📋 Complaints", "admin_complaints"),
        ("✏️ Update", "admin_update"),
        ("📊 Intelligence", "dashboard"),
        ("🚪 Logout", "logout"),
    ])


def require_role(role):
    if not st.session_state.authenticated or st.session_state.user_role != role:
        st.session_state.page = "login"
        st.rerun()


def citizen_portal():
    require_role("Citizen")
    citizen_nav()
    header("Citizen Grievance Portal", "PUBLIC GRIEVANCE SUBMISSION & TRACKING", "Citizen Access")
    complaints = get_citizen_complaints(st.session_state.user_email)
    counts = pd.Series([x.get("status", "Submitted") for x in complaints]).value_counts().to_dict() if complaints else {}
    st.markdown(f'<div class="card"><div class="card-title">Welcome</div><div class="card-text">Submit a grievance, track its progress and view privacy-safe intelligence about similar complaints.</div></div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    with a: render_kpi("MY COMPLAINTS", len(complaints), "Submitted by your account")
    with b: render_kpi("IN PROGRESS", int(sum(v for k,v in counts.items() if k not in ["Resolved","Rejected"])), "Currently active")
    with c: render_kpi("RESOLVED", int(counts.get("Resolved",0)), "Resolved complaints")
    with d: render_kpi("REJECTED", int(counts.get("Rejected",0)), "Closed without resolution")


def citizen_submit():
    require_role("Citizen")
    citizen_nav()
    header("Submit a Complaint", "CITIZEN GRIEVANCE REGISTRATION", "Citizen Access")
    c = mongo_collection()
    if c is None:
        st.error("The grievance service is currently unavailable. Please start MongoDB and try again.")
        return
    with st.form("complaint_form"):
        category = st.selectbox("Complaint Category", ["Water Supply","Sanitation","Electricity","Roads & Transport","Healthcare","Public Safety","Education","Other"])
        description = st.text_area("Complaint Description", height=150, placeholder="Describe the issue clearly, including what is happening and where.")
        state = st.text_input("State / Region", value="Telangana")
        district = st.text_input("District / City", value="Hyderabad")
        locality = st.text_input("Locality / Area", placeholder="Example: Quthbullapur")
        submitted = st.form_submit_button("SUBMIT COMPLAINT  →", type="primary", use_container_width=True)
    if submitted:
        if not description.strip() or not locality.strip():
            st.warning("Please provide the complaint description and locality.")
            return
        gid = next_grievance_id(c)
        now = utc_now()
        doc = {
            "grievance_id": gid,
            "citizen_email": st.session_state.user_email,
            "category": category,
            "description": description.strip(),
            "state": state.strip(),
            "dist_name": district.strip(),
            "location": locality.strip(),
            "department": "Not Assigned",
            "status": "Submitted",
            "priority_score": 0,
            "priority_reasons": [],
            "admin_remarks": "",
            "submitted_at": now,
            "updated_at": now,
            "status_history": [{"status":"Submitted","timestamp":now,"remark":"Complaint registered successfully."}],
        }
        try:
            c.insert_one(doc)
            st.success("Complaint Submitted Successfully")
            st.markdown(f"### Grievance ID: `{gid}`")
            st.info("Keep this grievance ID to track your complaint.")
        except Exception as exc:
            st.error(f"Unable to submit the complaint: {exc}")


def citizen_complaints():
    require_role("Citizen")
    citizen_nav()
    header("My Complaints", "YOUR GRIEVANCE RECORDS", "Citizen Access")
    docs = get_citizen_complaints(st.session_state.user_email)
    if not docs:
        st.info("You have not submitted any complaints yet.")
        return
    for doc in docs:
        with st.container(border=True):
            a,b,c = st.columns([2,2,1])
            with a:
                st.markdown(f"**{doc.get('grievance_id','')}**")
                st.caption(f"{doc.get('category','')} • {doc.get('location','')}")
            with b:
                st.write(f"Status: **{doc.get('status','Submitted')}**")
                st.write(f"Department: **{doc.get('department','Not Assigned')}**")
            with c:
                st.write(doc.get("updated_at", "")[:10])
            st.write(doc.get("description", ""))


def citizen_track():
    require_role("Citizen")
    citizen_nav()
    header("Track Complaint", "CURRENT STATUS & PRIVACY-SAFE INSIGHTS", "Citizen Access")
    gid = st.text_input("Grievance ID", placeholder="Example: PGI-2026-000005")
    if st.button("TRACK COMPLAINT  →", type="primary", use_container_width=True):
        doc = get_complaint(gid)
        if not doc:
            st.warning("Complaint not found.")
            return
        if doc.get("citizen_email") != st.session_state.user_email:
            st.error("You can only track complaints submitted from your account.")
            return
        st.subheader(f"{doc['grievance_id']}")
        a,b,c = st.columns(3)
        with a: st.metric("Status", doc.get("status", "Submitted"))
        with b: st.metric("Category", doc.get("category", "Not available"))
        with c: st.metric("Department", doc.get("department", "Not Assigned"))
        st.markdown("### Complaint Details")
        st.write(doc.get("description", ""))
        st.caption(f"Location: {doc.get('location','')}, {doc.get('dist_name','')}, {doc.get('state','')}")
        stats = similar_stats(doc)
        st.markdown("### Complaint Insights")
        a,b,c,d = st.columns(4)
        with a: st.metric("Similar Live Complaints", int(stats.get("count",0)))
        with b: st.metric("Resolved", f"{stats.get('resolved_percentage',0):.0f}%")
        with c: st.metric("Typical Resolution", format_resolution(stats.get("median_resolution_days")))
        _,_,hist_avg,hist_median,_,_ = historical_metrics()
        with d: st.metric("Historical Benchmark", format_resolution(hist_median if hist_median else hist_avg))
        if stats.get("status_distribution"):
            st.caption("Current status distribution among similar live complaints")
            st.bar_chart(pd.Series(stats["status_distribution"]))
        st.caption("Similar-complaint statistics are aggregate information and do not reveal other citizens' records. Historical resolution figures are context, not a prediction for this complaint.")


def citizen_notifications():
    require_role("Citizen")
    citizen_nav()
    header("Notifications", "GRIEVANCE UPDATES", "Citizen Access")
    docs = get_citizen_complaints(st.session_state.user_email)
    events=[]
    for d in docs:
        for h in d.get("status_history", []):
            events.append((h.get("timestamp",""), d.get("grievance_id",""), h.get("status",""), h.get("remark","")))
    events.sort(reverse=True)
    if not events:
        st.info("No notifications yet.")
    for ts,gid,status,remark in events:
        st.markdown(f"**{gid}** — {status}\n\n{remark}\n\n<small>{ts}</small>", unsafe_allow_html=True)
        st.divider()


def citizen_feedback():
    require_role("Citizen")
    citizen_nav()
    header("Feedback", "CITIZEN SATISFACTION", "Citizen Access")
    c=mongo_collection()
    docs=get_citizen_complaints(st.session_state.user_email)
    resolved=[d for d in docs if d.get("status")=="Resolved"]
    if not resolved:
        st.info("Feedback becomes available after a complaint is resolved.")
        return
    gid=st.selectbox("Resolved Complaint", [d["grievance_id"] for d in resolved])
    rating=st.slider("Satisfaction Rating",1,5,5)
    feedback=st.text_area("Feedback")
    if st.button("SUBMIT FEEDBACK  →", type="primary"):
        c.update_one({"grievance_id":gid},{"$set":{"feedback_rating":rating,"feedback":feedback,"feedback_at":utc_now()}})
        st.success("Thank you. Your feedback has been recorded.")


def admin_portal():
    require_role("Admin")
    admin_nav()
    header("Admin Grievance Portal", "COMPLAINT MANAGEMENT & DECISION INTELLIGENCE", "Admin Access")
    s=live_summary()
    a,b,c,d=st.columns(4)
    with a: render_kpi("LIVE COMPLAINTS", s.get("total",0), "Current operational workload")
    with b: render_kpi("PENDING", s.get("unresolved",0), "Submitted or under action")
    with c: render_kpi("RESOLVED", s.get("resolved",0), "Current live records")
    with d: render_kpi("REJECTED", s.get("rejected",0), "Current live records")
    st.markdown('<div class="section">ADMINISTRATIVE WORKSPACE</div>', unsafe_allow_html=True)
    p=get_priority_complaints() if get_priority_complaints else []
    if p:
        st.markdown('<div class="card"><div class="card-title">Priority Cases</div><div class="card-text">Rule-based operational prioritisation of unresolved complaints requiring attention.</div></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(p), use_container_width=True, hide_index=True)
    else:
        st.info("No priority cases currently identified.")


def admin_complaints():
    require_role("Admin")
    admin_nav()
    header("Complaint Management", "LIVE GRIEVANCE OPERATIONS", "Admin Access")
    c=mongo_collection()
    if c is None:
        st.error("MongoDB is unavailable.")
        return
    status=st.selectbox("Status", ["All","Submitted","Under Review","Assigned","In Progress","Resolved","Rejected"])
    category=st.text_input("Category filter")
    query={}
    if status!="All": query["status"]=status
    if category.strip(): query["category"]={"$regex":category.strip(),"$options":"i"}
    docs=list(c.find(query).sort("submitted_at",-1))
    if not docs:
        st.info("No complaints match the selected filters.")
        return
    rows=[]
    for d in docs:
        rows.append({"Grievance ID":d.get("grievance_id"),"Category":d.get("category"),"Location":d.get("location"),"Department":d.get("department"),"Status":d.get("status"),"Submitted":str(d.get("submitted_at",""))[:10]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"{len(docs)} live complaint(s)")


def admin_update():
    require_role("Admin")
    admin_nav()
    header("Update Complaint", "DEPARTMENT ASSIGNMENT & STATUS CONTROL", "Admin Access")
    c=mongo_collection()
    if c is None:
        st.error("MongoDB is unavailable.")
        return
    gid=st.text_input("Grievance ID")
    doc=get_complaint(gid) if gid else None
    if doc:
        st.info(f"{doc.get('category')} • {doc.get('location')} • Current status: {doc.get('status')}")
        st.write(doc.get("description",""))
        status=st.selectbox("Status", ["Submitted","Under Review","Assigned","In Progress","Resolved","Rejected"], index=["Submitted","Under Review","Assigned","In Progress","Resolved","Rejected"].index(doc.get("status","Submitted")))
        department=st.text_input("Assigned Department", value=doc.get("department","Not Assigned"))
        remarks=st.text_area("Administrative Remarks", value=doc.get("admin_remarks",""))
        if st.button("SAVE UPDATE  →", type="primary", use_container_width=True):
            now=utc_now()
            history=doc.get("status_history",[])
            history.append({"status":status,"timestamp":now,"remark":remarks.strip()})
            c.update_one({"grievance_id":gid.strip().upper()},{"$set":{"status":status,"department":department.strip() or "Not Assigned","admin_remarks":remarks.strip(),"updated_at":now,"status_history":history}})
            st.success("Complaint updated successfully.")
            st.rerun()
    elif gid:
        st.warning("Complaint not found.")


def intelligence_dashboard():
    require_role("Admin")
    admin_nav()
    header("Public Grievance Intelligence", "LIVE OPERATIONS + HISTORICAL CONTEXT + AI DECISION SUPPORT", "Admin Intelligence")
    total,resolved,avg,median,orgs,states=historical_metrics()
    live=live_summary()
    st.markdown('<div class="title">Grievance Intelligence Dashboard</div><div class="subtitle">Current live complaints are compared with historical grievance patterns to provide operational context, concentration signals and evidence-based administrative decision support.</div>', unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    with a: render_kpi("LIVE COMPLAINTS", live.get("total",0), "Current operational workload")
    with b: render_kpi("HISTORICAL RECORDS", f"{total:,}", "Historical context")
    with c: render_kpi("HISTORICAL MEDIAN", format_resolution(median), "Historical resolution context")
    with d: render_kpi("HISTORICAL AVERAGE", format_resolution(avg), "Historical resolution context")

    st.markdown('<div class="section">CURRENT LIVE ANALYTICS</div>', unsafe_allow_html=True)
    if get_complaints_by_category:
        try:
            cat=get_complaints_by_category()
            if cat:
                c1,c2=st.columns(2)
                with c1:
                    st.plotly_chart(px.bar(pd.DataFrame(cat.items(),columns=["Category","Complaints"]).sort_values("Complaints",ascending=False),x="Category",y="Complaints",title="Live Complaints by Category"),use_container_width=True,config={"displayModeBar":False})
                with c2:
                    status=get_status_distribution() if get_status_distribution else {}
                    if status:
                        st.plotly_chart(px.pie(pd.DataFrame(status.items(),columns=["Status","Count"]),names="Status",values="Count",title="Live Status Distribution"),use_container_width=True,config={"displayModeBar":False})
        except Exception as exc:
            st.warning(f"Live analytics unavailable: {exc}")
    else:
        st.info("Live intelligence module is unavailable.")

    st.markdown('<div class="section">HISTORICAL ANALYTICS</div>', unsafe_allow_html=True)
    if not historical.empty:
        h1,h2=st.columns(2)
        if "state" in historical.columns:
            state_counts=historical["state"].fillna("Unknown").astype(str).str.strip().value_counts().head(12).sort_values()
            with h1:
                st.plotly_chart(px.bar(state_counts,orientation="h",title="Historical Grievance Concentration by Region",labels={"value":"Grievances","state":"Region"}),use_container_width=True,config={"displayModeBar":False})
        if "resolution_days" in historical.columns:
            with h2:
                st.plotly_chart(px.histogram(historical.dropna(subset=["resolution_days"]),x="resolution_days",nbins=35,title="Historical Resolution-Time Distribution",labels={"resolution_days":"Resolution days"}),use_container_width=True,config={"displayModeBar":False})

    st.markdown('<div class="section">AI GRIEVANCE INTELLIGENCE</div>', unsafe_allow_html=True)
    question=st.text_area("Ask the administrator intelligence assistant",height=110,placeholder="Example: What are the major current grievance issues and how do they compare with historical complaints?")
    if st.button("GENERATE INTELLIGENCE REPORT  →",type="primary",use_container_width=True):
        if not question.strip():
            st.warning("Enter an administrator question first.")
        elif ask_ai_grievance_intelligence is None:
            st.error("The AI intelligence engine is unavailable. Check src/ai_grievance_engine.py.")
        else:
            with st.spinner("Preparing evidence-based intelligence report..."):
                result=ask_ai_grievance_intelligence(question.strip())
            st.markdown('<div class="insight">',unsafe_allow_html=True)
            st.markdown(result.get("answer","No report generated."))
            st.markdown('</div>',unsafe_allow_html=True)


page=st.session_state.page
if page=="landing": landing_page = login_page()
elif page=="login": auth_page()
elif page=="citizen_portal": citizen_portal()
elif page=="citizen_submit": citizen_submit()
elif page=="citizen_complaints": citizen_complaints()
elif page=="citizen_track": citizen_track()
elif page=="citizen_notifications": citizen_notifications()
elif page=="citizen_feedback": citizen_feedback()
elif page=="admin_portal": admin_portal()
elif page=="admin_complaints": admin_complaints()
elif page=="admin_update": admin_update()
elif page=="dashboard": intelligence_dashboard()
else:
    st.session_state.page="landing"
    st.rerun()
