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

# Main Dashboard
if submit_btn:
    with st.spinner(f"Requesting map-matching & reconstruction for {selected_trip} from {api_url}..."):
        try:
            response = requests.post(
                f"{api_url}/predict",
                json={"trip_id": selected_trip},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                metrics = data["metrics"]

                st.success(f"Successfully reconstructed route for **{selected_trip}**! (Status: {data.get('status')})")

                # Metric Cards
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Top-1 Accuracy", f"{metrics['top_1_accuracy'] * 100:.2f}%")
                col2.metric("Macro F1-Score", f"{metrics['macro_f1']:.4f}")
                col3.metric("Route Continuity", f"{metrics['route_continuity'] * 100:.2f}%")
                col4.metric("Inference Latency", f"{metrics['inference_time_ms']:.1f} ms")

                st.markdown("---")
                st.subheader("Interactive Folium Geographical Map")

                gps_points = data.get("gps_trajectory", [])
                pred_coords = data.get("predicted_route_coords", [])
                gt_coords = data.get("ground_truth_route_coords", [])

                # Determine map center
                if gps_points:
                    center_lat = gps_points[0]["latitude"]
                    center_lon = gps_points[0]["longitude"]
                elif pred_coords:
                    center_lat, center_lon = pred_coords[0][0], pred_coords[0][1]
                else:
                    center_lat, center_lon = 12.9716, 77.5946

                # Create folium map
                m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="CartoDB positron")

                # 1. Layer for GPS observations (blue circles)
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

                # 2. Layer for Ground Truth route (green line)
                if gt_coords:
                    gt_group = folium.FeatureGroup(name="Reference Ground Truth (Green)")
                    folium.PolyLine(
                        locations=gt_coords,
                        color="green",
                        weight=5,
                        opacity=0.7,
                        tooltip="Ground Truth Route",
                    ).add_to(gt_group)
                    gt_group.add_to(m)

                # 3. Layer for Predicted Reconstructed route (red dashed line)
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

                # Add Layer Control
                folium.LayerControl().add_to(m)

                # Render map in Streamlit
                st_folium(m, width="100%", height=600)

                # Sequence inspection expander
                with st.expander("📋 View Predicted Road Arc IDs"):
                    st.write(data.get("predicted_arcs", []))

            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to Laptop 1 at `{api_url}`. Please verify the network connection and backend server.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
else:
    st.info("👈 Select an evaluation trip from the sidebar and click **'Reconstruct Route'** to run live inference.")
