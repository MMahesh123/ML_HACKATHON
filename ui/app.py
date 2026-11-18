"""Streamlit Evaluator Interface (Laptop 2 Client)."""

import os
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

# Streamlit page layout configuration
st.set_page_config(
    page_title="PathSnap — Evaluator Interface",
    page_icon="🗺️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="st-"] { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(255, 0, 170, 0.18), transparent 40%),
        radial-gradient(circle at 85% 0%, rgba(0, 220, 255, 0.16), transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(130, 0, 255, 0.18), transparent 50%),
        #0a0a14;
    background-attachment: fixed;
}

@keyframes glowpulse {
    0%, 100% { text-shadow: 0 0 18px rgba(255,0,200,0.55), 0 0 36px rgba(0,220,255,0.35); }
    50% { text-shadow: 0 0 30px rgba(0,220,255,0.65), 0 0 50px rgba(255,0,200,0.45); }
}
@keyframes gradientflow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes floatup {
    0% { opacity: 0; transform: translateY(14px); }
    100% { opacity: 1; transform: translateY(0); }
}

.pathsnap-hero {
    padding: 2.2rem 2rem 1.6rem 2rem;
    border-radius: 22px;
    margin-bottom: 1.4rem;
    background: linear-gradient(120deg, rgba(255,0,170,0.14), rgba(0,220,255,0.10), rgba(130,0,255,0.14));
    background-size: 200% 200%;
    animation: gradientflow 10s ease infinite, floatup 0.6s ease-out;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
}
.pathsnap-hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #ff2fd1, #00e5ff, #a855f7, #ff2fd1);
    background-size: 300% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: gradientflow 6s linear infinite, glowpulse 3s ease-in-out infinite;
}
.pathsnap-hero .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9ad4ff;
    opacity: 0.85;
    margin-top: 0.35rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a, #14091f 60%, #0a0a14);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #e8e6ff !important; }

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    border: 1px solid rgba(0,229,255,0.25);
    border-radius: 16px;
    padding: 0.9rem 0.8rem 0.6rem 0.8rem;
    box-shadow: 0 0 22px rgba(0,229,255,0.08), inset 0 0 20px rgba(255,0,200,0.04);
    animation: floatup 0.5s ease-out;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 28px rgba(255,0,200,0.25), 0 0 14px rgba(0,229,255,0.2);
}
[data-testid="stMetricValue"] {
    background: linear-gradient(90deg, #00e5ff, #ff2fd1);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-weight: 700;
}

div.stButton > button, section[data-testid="stSidebar"] button {
    background: linear-gradient(90deg, #ff2fd1, #a855f7, #00e5ff);
    background-size: 200% auto;
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    letter-spacing: 0.03em;
    transition: all 0.25s ease;
    box-shadow: 0 0 18px rgba(168,85,247,0.4);
}
div.stButton > button:hover, section[data-testid="stSidebar"] button:hover {
    background-position: right center;
    box-shadow: 0 0 28px rgba(255,47,209,0.6), 0 0 14px rgba(0,229,255,0.5);
    transform: scale(1.02);
}

[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(6px);
}

iframe { border-radius: 16px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pathsnap-hero">
  <h1>🗺️ PathSnap</h1>
  <div class="tag">Distributed Map-Matching &amp; Route Reconstruction Engine</div>
  <div class="tag">PES University · UE24CS352A Machine Learning Hackathon</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Server Connection (Laptop 1)")
api_url = st.sidebar.text_input(
    "API Endpoint URL",
    value=os.getenv("API_URL", "http://localhost:8000"),
    help="Enter the IP address and port of Laptop 1 running the FastAPI server (e.g. http://192.168.1.50:8000)"
)

# Fetch available evaluation trips
@st.cache_data(ttl=10)
def fetch_trips(base_url: str):
    try:
        res = requests.get(f"{base_url}/trips", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return [f"trip_{i:02d}" for i in range(1, 11)]

trips = fetch_trips(api_url)

st.sidebar.subheader("🎯 Evaluation Selection")
selected_trip = st.sidebar.selectbox("Choose Evaluation Trip:", trips)
submit_btn = st.sidebar.button("🚀 Reconstruct Route", type="primary")

# Results are stashed in session_state rather than rendered only inside
# `if submit_btn:` - the interactive Folium map (st_folium) triggers its own
# rerun right after mounting to report its state back to Python, and on that
# follow-up rerun `submit_btn` is False again. Rendering straight off
# session_state means the result survives that rerun instead of flickering
# back to the placeholder hint.
if submit_btn:
    with st.spinner(f"Requesting map-matching & reconstruction for {selected_trip} from {api_url}..."):
        try:
            response = requests.post(
                f"{api_url}/predict",
                json={"trip_id": selected_trip},
                timeout=30,
            )
            if response.status_code == 200:
                st.session_state["last_result"] = {"trip_id": selected_trip, "data": response.json(), "error": None}
            else:
                st.session_state["last_result"] = {"trip_id": selected_trip, "data": None, "error": f"Error {response.status_code}: {response.text}"}
        except requests.exceptions.ConnectionError:
            st.session_state["last_result"] = {
                "trip_id": selected_trip, "data": None,
                "error": f"Could not connect to Laptop 1 at `{api_url}`. Please verify the network connection and backend server.",
            }
        except Exception as e:
            st.session_state["last_result"] = {"trip_id": selected_trip, "data": None, "error": f"Unexpected error: {e}"}

result = st.session_state.get("last_result")

if result is None:
    st.info("👈 Select an evaluation trip from the sidebar and click **'Reconstruct Route'** to run live inference.")
elif result["error"] is not None:
    st.error(result["error"])
else:
    data = result["data"]
    metrics = data["metrics"]

    st.markdown(f"""
    <div style="padding:0.9rem 1.2rem;border-radius:14px;margin-bottom:0.8rem;
         background:linear-gradient(90deg, rgba(0,255,170,0.12), rgba(0,229,255,0.08));
         border:1px solid rgba(0,255,170,0.35); animation: floatup 0.4s ease-out;">
      ✅ <b>Route reconstructed for {result['trip_id']}</b> &nbsp;·&nbsp; status: <code>{data.get('status')}</code>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top-1 Accuracy", f"{metrics['top_1_accuracy'] * 100:.2f}%")
    macro_f1 = metrics.get("macro_f1")
    col2.metric("Macro F1-Score", f"{macro_f1:.4f}" if macro_f1 is not None else "N/A")
    col3.metric("Route Continuity", f"{metrics['route_continuity'] * 100:.2f}%")
    col4.metric("Inference Latency", f"{metrics['inference_time_ms']:.1f} ms")
    st.caption(f"Model: {metrics.get('model_name', 'unknown')}")

    st.markdown("---")
    st.markdown("### 🛰️ Interactive Folium Geographical Map")

    gps_points = data.get("gps_trajectory", [])
    pred_coords = data.get("predicted_route_coords", [])
    # Ground-truth route is intentionally never sent by the API - it stays
    # server-side so the UI cannot leak evaluation labels.

    if gps_points:
        center_lat = gps_points[0]["latitude"]
        center_lon = gps_points[0]["longitude"]
    elif pred_coords:
        center_lat, center_lon = pred_coords[0][0], pred_coords[0][1]
    else:
        center_lat, center_lon = 12.9716, 77.5946

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    gps_group = folium.FeatureGroup(name="Observed GPS Trajectory (Noisy)")
    for pt in gps_points:
        folium.CircleMarker(
            location=[pt["latitude"], pt["longitude"]],
            radius=3,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup=f"GPS Time: {pt.get('timestamp')}",
        ).add_to(gps_group)
    gps_group.add_to(m)

    if pred_coords:
        pred_group = folium.FeatureGroup(name="PathSnap Predicted Route (Red/Orange)")
        folium.PolyLine(
            locations=pred_coords,
            color="red",
            weight=3,
            dash_array="5, 5",
            tooltip="Reconstructed PathSnap Route",
        ).add_to(pred_group)
        pred_group.add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=600, key="pathsnap_map")

    with st.expander("📋 View Predicted Road Arc IDs"):
        st.write(data.get("predicted_arcs", []))
