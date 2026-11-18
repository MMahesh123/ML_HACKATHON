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

st.title("🗺️ PathSnap: Distributed Route Reconstruction")
st.caption("PES University — UE24CS352A Machine Learning Hackathon")

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

    st.success(f"Successfully reconstructed route for **{result['trip_id']}**! (Status: {data.get('status')})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top-1 Accuracy", f"{metrics['top_1_accuracy'] * 100:.2f}%")
    macro_f1 = metrics.get("macro_f1")
    col2.metric("Macro F1-Score", f"{macro_f1:.4f}" if macro_f1 is not None else "N/A")
    col3.metric("Route Continuity", f"{metrics['route_continuity'] * 100:.2f}%")
    col4.metric("Inference Latency", f"{metrics['inference_time_ms']:.1f} ms")
    st.caption(f"Model: {metrics.get('model_name', 'unknown')}")

    st.markdown("---")
    st.subheader("Interactive Folium Geographical Map")

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
