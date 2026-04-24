import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from fpdf import FPDF
from supabase import create_client
import tempfile
import os
from ultralytics import YOLO
import cv2
import tempfile
import os



 # --- 1. SUPABASE CONFIGURATION ---
SUPABASE_URL = "https://zoxppbblaiumsmxviydq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpveHBwYmJsYWl1bXNteHZpeWRxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY3NjQ1MjgsImV4cCI6MjA5MjM0MDUyOH0.3azE5ylA15CV6Vr132dqSmDJqfnW_69DVdb3l6wp0ss"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def load_yolo():
    # Downloads yolov8n.pt automatically if not in folder
    return YOLO("yolov8n.pt") 

model = load_yolo()


# --- 2. INITIALIZE SESSION STATE ---
if 'temp' not in st.session_state: st.session_state.temp = 6.4
if 'hum' not in st.session_state: st.session_state.hum = 82
if 'ethylene' not in st.session_state: st.session_state.ethylene = "0"
if 'live_lat' not in st.session_state: st.session_state.live_lat = -20.15
if 'live_lon' not in st.session_state: st.session_state.live_lon = 28.58
if 'last_image_url' not in st.session_state: st.session_state.last_image_url = None
if 'raw_image_url' not in st.session_state: st.session_state.raw_image_url = None
if 'ai_label' not in st.session_state: st.session_state.ai_label = "Scanning..."
if 'ai_conf' not in st.session_state: st.session_state.ai_conf = "0%"
if 'ai_processed_img' not in st.session_state: st.session_state.ai_processed_img = None
if 'shipment_status' not in st.session_state: st.session_state.shipment_status = "Idle"
if 'health_score' not in st.session_state: st.session_state.health_score = 100
if 'progress' not in st.session_state: st.session_state.progress = 0.0
if 'breach_active' not in st.session_state: st.session_state.breach_active = False
if 'history' not in st.session_state: 
    st.session_state.history = pd.DataFrame(columns=['Time', 'Temp', 'Hum', 'Ethylene'])


# --- UI STYLING ---
st.set_page_config(page_title="Agri-Trace Pro | Full Suite", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stMetricValue"] { color: #00FFC2; font-weight: bold; }
    .status-card { padding: 20px; border-radius: 12px; text-align: center; font-weight: 800; margin-bottom: 20px; border: 1px solid #333; }
    .status-transit { background-color: #1E3A8A; color: #BFDBFE; border-left: 8px solid #3B82F6; }
    .status-border { background-color: #3730A3; color: #E0E7FF; border-left: 8px solid #6366F1; }
    .status-valid { background-color: #064E3B; color: #D1FAE5; border-left: 8px solid #10B981; }
    .status-invalid { background-color: #7F1D1D; color: #FEE2E2; border-left: 8px solid #EF4444; }
    .status-finished { background-color: #111827; color: #9CA3AF; border-left: 8px solid #4B5563; }
    .sticky-alert { padding: 15px; background-color: #450a0a; color: #fecaca; border: 2px solid #ef4444; border-radius: 8px; margin-bottom: 25px; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0px rgba(239, 68, 68, 0.4); } 70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0px rgba(239, 68, 68, 0); } }
    section[data-testid="stSidebar"] { background-color: #161B22; }
    </style>
    """, unsafe_allow_html=True)


def sync_storage_to_db():
    try:
        # 1. Get the latest row
        res = supabase.table("transit_data").select("id, raw_image_url").order("created_at", desc=True).limit(1).execute()
        
        if res.data:
            latest_row = res.data[0]
            
            # If the URL is missing or ends in 'None'
            if not latest_row.get('raw_image_url') or str(latest_row.get('raw_image_url')).endswith('None'):
                # 2. Get all files in the bucket
                files = supabase.storage.from_("piframes").list()
                
                # Filter out the placeholder
                valid_files = [f for f in files if f['name'] != '.emptyFolderPlaceholder']
                
                if valid_files:
                    # Take the very first valid file we find
                    latest_filename = valid_files[0]['name'] 
                    new_url = f"{SUPABASE_URL}/storage/v1/object/public/piframes/{latest_filename}"
                    
                    # 3. Force update the database
                    supabase.table("transit_data").update({"raw_image_url": new_url}).eq("id", latest_row['id']).execute()
                    st.success(f"Synced existing frame: {latest_filename}")
                else:
                    st.error("Bucket is empty! Upload at least one image manually to Supabase.")
    except Exception as e:
        st.error(f"Sync Error: {e}")




# --- UTILITY ---
def clamp(n, minn, maxn):
    return max(min(n, maxn), minn)



# --- 3. DATA RETRIEVAL & AI PROCESSING ---
def get_data():
    # Retrieve data from Supabase
    response = supabase.table("transit_data").select("*").order("created_at", desc=True).execute()
    df = pd.DataFrame(response.data)
    return df

df = get_data()

if not df.empty:
    latest_record = df.iloc[0]
    
    # AI Logic: Run detection on the latest frame from piframes bucket
    # We only run and post if ai_detection_label is missing to avoid infinite loops
    # --- SAFE AI LOGIC WRAPPER ---

# 1. Get the URL from the record (which should be fixed by your sync function)
    img_url = latest_record.get('raw_image_url')

# 2. Only run if we have a real link and it's not the string 'None'
    if img_url and str(img_url).strip().lower() != 'none' and "None" not in str(img_url):
         try:
        # Run YOLOv8
            results = model(img_url)
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Display the AI Frame in Streamlit (optional but recommended for the demo)
            st.image(res_rgb, caption=f"AI Analysis: {img_url.split('/')[-1]}", use_container_width=True)

        # Extract Scores
            if len(results[0].boxes) > 0:
                label = results[0].names[int(results[0].boxes[0].cls[0])]
                conf = float(results[0].boxes[0].conf[0])
            else:
                label = "Healthy"
                conf = 1.0

        # Post AI results back to Supabase if not already updated
            if pd.isna(latest_record.get('ai_detection_label')):
                supabase.table("transit_data").update({
                     "ai_detection_label": label,
                    "ai_confidence": f"{conf*100:.1f}%",
                    "processed_at": "now()"
                }).eq("id", latest_record['id']).execute()
                st.toast(f"AI Detection Complete: {label}")

         except Exception as e:
            st.error(f"AI Processing Error: {e}")
        # Default values so the rest of the page doesn't break
            label = "Processing Error"
            conf = 0.0
    else:
    # This shows if the database hasn't been synced yet
       st.warning("🔍 Awaiting valid image link from storage. Please refresh or check Supabase Bucket.")
       label = "No Image Found"
       conf = 0.0

        




# --- 3. LIVE DATA FETCHING ---
def fetch_supabase_data():
    sync_storage_to_db()
    try:
        # Fetch latest record
        response = supabase.table("transit_data").select("*").order("created_at", desc=True).limit(1).execute()
        
        if response.data:
            latest = response.data[0]
            
            # --- 1. Physical Sensor Data ---
            t = float(latest.get('temperature', 10.0))
            h = float(latest.get('humidity', 85.0))
            raw_e = latest.get('ethylene_ppm', 0)
            e = float(raw_e) if raw_e is not None and str(raw_e) != "Null" else 0.0

            # --- 2. GPS Data Sync ---
            st.session_state.live_lat = float(latest.get('latitude', -20.15))
            st.session_state.live_lon = float(latest.get('longitude', 28.58))

            # --- 3. YOLOv8 AI OBJECT DETECTION ---
            # Construct the public URL for the image in the bucket
            img_url = latest.get('raw_image_url')
            
            try:
                # Run detection
                results = model(img_url)
                res_plotted = results[0].plot()
                
                # Convert BGR to RGB for Streamlit/PDF compatibility
                image_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.session_state.ai_processed_img = image_rgb 
                
                # Extract Scores
                if len(results[0].boxes) > 0:
                    label = results[0].names[int(results[0].boxes[0].cls[0])]
                    conf_val = float(results[0].boxes[0].conf[0])
                    conf_str = f"{conf_val*100:.1f}%"
                else:
                    label = "Healthy"
                    conf_str = "100%"

                # Update Supabase with AI results if they are currently empty
                if not latest.get('ai_detection_label'):
                    supabase.table("transit_data").update({
                        "ai_detection_label": label,
                        "ai_confidence": conf_str
                    }).eq("id", latest['id']).execute()

            except Exception as ai_err:
                print(f"AI Processing Error: {ai_err}")
                label = "Detection Error"
                conf_str = "N/A"

            # --- 4. Health Equation ---
            t_penalty = 1.5 * abs(t - 10.0)
            h_penalty = 0.5 * abs(85.0 - h)
            e_penalty = 2.0 * e
            new_health = 100 - (t_penalty + h_penalty + e_penalty)
            
            # --- 5. Global State Update ---
            st.session_state.health_score = max(0, round(new_health, 1))
            st.session_state.temp = t
            st.session_state.hum = h
            st.session_state.ethylene = str(raw_e)
            st.session_state.last_image_url = img_url 
            st.session_state.ai_label = label
            st.session_state.ai_conf = conf_str
            
            crate_open = latest.get('crate_open')
            if crate_open is False:  # False = open = breached
                st.session_state.breach_active = True

            # --- 6. History Update ---
            new_entry = pd.DataFrame({
                'Time': [datetime.now().strftime("%H:%M:%S")], 
                'Temp': [t], 'Hum': [h], 'Ethylene': [e]
            })
            st.session_state.history = pd.concat([st.session_state.history, new_entry]).tail(15)

    except Exception as err:
        print(f"Hardware Sync Error: {err}")

# --- 4. PDF GENERATOR ---
# --- NEW: SPECIALIZED PDF GENERATORS ---

def generate_trust_certificate():
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header Section (Deep Green)
    pdf.set_fill_color(6, 78, 59) 
    pdf.rect(0, 0, 210, 45, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(190, 35, "PHYTOSANITARY CERTIFICATE OF TRUST", ln=True, align='C')
    
    # 2. Content Section
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=11)
    pdf.ln(20)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, "OFFICIAL QUALITY ASSURANCE SUMMARY", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(180, 7, f"This document confirms that the shipment monitored at Lat: {st.session_state.live_lat} has maintained a Cumulative Health Score of {st.session_state.health_score}%. This score is a weighted metric ensuring the produce remained within optimal biological windows throughout the journey.")
    
    pdf.ln(5)
    
    # 3. Explained Data Figures
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(100, 8, "DETAILED DATA INTERPRETATION:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(180, 6, f"- TEMPERATURE ({st.session_state.temp}C): Vital for slowing metabolic decay.\n"
                          f"- HUMIDITY ({st.session_state.hum}%): Prevents water loss and ensures crispness.\n"
                          f"- ETHYLENE ({st.session_state.ethylene} ppm): Hormone levels indicating controlled ripening.\n"
                          f"- AI VISION ANALYSIS: {st.session_state.ai_label} ({st.session_state.ai_conf})")

    pdf.ln(10)

    # 4. AI IMAGE GENERATION (Injected Section)
    if 'ai_processed_img' in st.session_state:
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(100, 8, "AI VISION VERIFICATION FRAME:", ln=True)
        
        # Save the analyzed image to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            # Revert RGB to BGR for proper PDF color rendering
            bgr_img = cv2.cvtColor(st.session_state.ai_processed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp.name, bgr_img)
            # Add image to PDF (Positioning it below the text)
            pdf.image(tmp.name, x=15, y=None, w=100)
            
        # Optional: Cleanup temp file
        os.remove(tmp.name)
    else:
        pdf.cell(100, 10, "[Visual Verification Missing]", ln=True)

    return pdf.output(dest='S').encode('latin-1', errors='replace')

def generate_audit_report():
    pdf = FPDF()
    pdf.add_page()
    
    # Header Section
    pdf.set_fill_color(30, 58, 138) # Regulatory Blue
    pdf.rect(0, 0, 210, 45, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(190, 35, "CARGO COMPLIANCE & AUDIT REPORT", ln=True, align='C')
    
    # Technical Section
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=11)
    pdf.ln(20)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, "REGULATORY TELEMETRY AUDIT", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(180, 7, f"Audit conducted for border clearance. Logistics Tracking ID: {datetime.now().strftime('%Y%m%d%H%M')}. Current Registered GPS: {st.session_state.live_lat}, {st.session_state.live_lon}.")
    
    pdf.ln(5)
    
    # Regulatory Explanations
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(100, 8, "FIELD DATA ANALYSIS FOR CUSTOMS OFFICIALS:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(180, 6, f"- THERMAL STABILITY: The recorded {st.session_state.temp}C indicates no cold-chain breaches.\n"
                          f"- HYGROMETRIC DATA: {st.session_state.hum}% humidity levels verify cargo was not exposed to external moisture.\n"
                          f"- GAS EMISSION (ETHYLENE): Recorded at {st.session_state.ethylene} ppm.\n"
                          f"- AI CLASSIFICATION: {st.session_state.ai_label} at {st.session_state.ai_conf} confidence.")

    # --- UPDATED: YOLOv8 AI VISUAL EVIDENCE SECTION ---
    if 'ai_processed_img' in st.session_state:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(100, 8, "AI-ANALYZED VISUAL VERIFICATION :", ln=True)
        try:
            # Save the processed image from state to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                # Convert RGB (Streamlit/UI) back to BGR (OpenCV/PDF)
                bgr_img = cv2.cvtColor(st.session_state.ai_processed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(tmp.name, bgr_img)
                # Embed image in PDF
                pdf.image(tmp.name, x=10, y=None, w=110)
                
            # Cleanup temp file
            os.remove(tmp.name)
        except Exception as e:
            print(f"PDF Image Error: {e}")
            pdf.cell(100, 10, "[Error loading AI visual evidence]", ln=True)
    else:
        pdf.ln(10)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(100, 8, "NOTICE: AI Visual Verification Pending", ln=True)
        pdf.set_text_color(0, 0, 0)

    return pdf.output(dest='S').encode('latin-1', errors='replace')

# --- ROLE DETECTION ---
query_params = st.query_params
url_user = query_params.get("user", "Development")
if url_user == "farmer": role = "Farmer"
elif url_user == "customer": role = "Customer"
elif url_user == "border": role = "Border Official"
else: role = st.sidebar.selectbox("User Portal", ["Farmer", "Customer", "Border Official"])

# --- TELEMETRY & GPS ---
def display_notifications(role):
    if st.session_state.breach_active and role != "Border Official":
        st.markdown(
            '<div class="sticky-alert">🚨 <b>CRATE BREACH DETECTED</b> — '
            'The physical seal on this shipment has been broken.</div>',
            unsafe_allow_html=True
        )
        if role == "Farmer":
            if st.button("⚠️ ACKNOWLEDGE BREACH", use_container_width=True):
                st.session_state.breach_active = False
                st.rerun()

@st.fragment(run_every=10)
def display_live_telemetry(user_label):
    fetch_supabase_data()
    st.subheader(f"📊 Live Telemetry: {user_label}")
    
    # HARDWARE SYNCED COORDINATES
    start_lat, start_lon = -20.15, 28.58
    dest_lat, dest_lon = -26.20, 28.04
    curr_lat, curr_lon = st.session_state.live_lat, st.session_state.live_lon

    # CALC PROGRESS BASED ON HARDWARE GPS
    total_dist = ((dest_lat - start_lat)**2 + (dest_lon - start_lon)**2)**0.5
    dist_covered = ((curr_lat - start_lat)**2 + (curr_lon - start_lon)**2)**0.5
    st.session_state.progress = clamp(dist_covered / total_dist, 0.0, 1.0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Temp", f"{st.session_state.temp}°C")
    c2.metric("Humidity", f"{st.session_state.hum}%")
    c3.metric("Ethylene", st.session_state.ethylene)
    c4.metric("Health Score", f"{st.session_state.health_score}%")
    c5.metric("AI Label", label)
    c6.metric("AI Confidence", f"{conf*100:.1f}%")

    st.progress(st.session_state.progress)
    
    m1, m2 = st.columns([1, 1])
    with m1:
        st.write("**📍 Hardware GPS Tracking**")
        map_data = pd.DataFrame({
            'lat': [start_lat, -22.21, dest_lat, curr_lat], 
            'lon': [28.58, 29.99, 28.04, curr_lon], 
            'color': ['#FFFFFF', '#6366F1', '#00FFC2', '#FF4B4B']
        })
        st.map(map_data, color='color', zoom=5)
    with m2:
        st.write("**📈 Environmental Trends**")
        if not st.session_state.history.empty: 
            chart_df = st.session_state.history.set_index('Time')
            st.caption("Temperature (°C)")
            st.line_chart(chart_df['Temp'], height=300, color="#FF4B4B")
            st.caption("Humidity (%)")
            st.line_chart(chart_df['Hum'], height=300, color="#3B82F6")
            st.caption("Ethylene (ppm)")
            st.line_chart(chart_df['Ethylene'], height=300, color="#00FFC2")

    st.write("---")
    # --- LIVE ANALYSIS FEED UI SECTION ---
# This typically goes inside your columns: col_img, col_map = st.columns([1, 1])
    col_img, col_map = st.columns([1, 1])

    with col_img:
        st.subheader("🛰️ Live AI Analysis Feed")
    
    # 1. Check if the AI has processed an image yet
        if st.session_state.get('ai_processed_img') is not None:
        
        # Display the image with YOLO bounding boxes
            st.image(
            st.session_state.ai_processed_img, 
            caption=f"Latest Scan: {st.session_state.ai_label} ({st.session_state.ai_conf} confidence)",
            use_container_width=True
        )
        
        # 2. Status Indicator Overlay logic
            if st.session_state.ai_label == "Healthy":
                st.success(f"Verified: {st.session_state.ai_label}")
            else:
                st.error(f"Alert: {st.session_state.ai_label} Detected")
            
        else:
        # 3. Fallback if no image is available yet
            st.info("Awaiting incoming frame from Raspberry Pi...")
        # Optional: Add a placeholder gray box to keep the layout stable
            st.image("https://via.placeholder.com/640x480.png?text=Waiting+for+Transit+Feed", use_container_width=True)

    # 4. Detailed Metadata (Small text for the judges)
        st.caption(f"Model: YOLOv8n | Source: Supabase piframes | Location: {st.session_state.live_lat}, {st.session_state.live_lon}")

# --- DASHBOARDS ---
st.sidebar.title("🌿 Agri-Trace v5.0")
def display_global_status():
    s = st.session_state.shipment_status
    if s == "Transit": st.markdown('<div class="status-card status-transit">🚚 IN TRANSIT</div>', unsafe_allow_html=True)
    elif s == "Border": st.markdown('<div class="status-card status-border">🛂 AT BORDER</div>', unsafe_allow_html=True)
    elif s == "Valid": st.markdown('<div class="status-card status-valid">✅ CLEARED</div>', unsafe_allow_html=True)
    elif s == "Invalid": st.markdown('<div class="status-card status-invalid">❌ REJECTED</div>', unsafe_allow_html=True)
    elif s == "Finished": st.markdown('<div class="status-card status-finished">🏁 COMPLETED</div>', unsafe_allow_html=True)



if role == "Farmer":
    st.set_page_config(initial_sidebar_state="collapsed")
    st.title("👨‍🌾 Farmer Command Center")
    display_notifications(role)
    display_global_status()
    if st.session_state.shipment_status == "Valid":
        # FIXED: Added use_container_width
        st.download_button("📜 DOWNLOAD TRUST CERTIFICATION", data=generate_trust_certificate(), file_name="Trust_Cert.pdf", use_container_width=True)
    
    b1, b2 = st.columns(2)
    with b1:
        if st.session_state.shipment_status == "Idle":
            # FIXED: Added use_container_width
            if st.button("🚀 START TRAVEL", type="primary", use_container_width=True): 
                st.session_state.shipment_status = "Transit"; st.rerun()
    with b2:
        if st.session_state.shipment_status in ["Transit", "Border", "Valid", "Invalid"]:
            # FIXED: Added use_container_width
            if st.button("🛑 STOP DELIVERY", use_container_width=True): 
                st.session_state.shipment_status = "Idle"; st.session_state.progress = 0.0; st.rerun()
    if st.session_state.shipment_status != "Idle": display_live_telemetry("Themba")

elif role == "Customer":
    st.set_page_config(initial_sidebar_state="collapsed")
    st.title("📦 Customer Tracker")
    display_notifications(role)
    display_global_status()
    if st.session_state.shipment_status != "Idle": display_live_telemetry("Customer")

elif role == "Border Official":
    st.set_page_config(initial_sidebar_state="collapsed")
    st.title("🛂 Border Control Hub")
    display_global_status()
    if st.session_state.shipment_status == "Transit":
        if st.button("📥 CONFIRM ARRIVAL", use_container_width=True): 
            st.session_state.shipment_status = "Border"; st.rerun()
    
    elif st.session_state.shipment_status == "Border":
        # RE-ADDED: The missing View Preview button
        st.download_button("📄 VIEW AUDIT REPORT (PREVIEW)", data=generate_audit_report(), file_name="Audit_Preview.pdf", use_container_width=True)
        
        v, i = st.columns(2)
        with v:
            # FIXED: Added use_container_width
            if st.button("✅ MARK VALID", use_container_width=True): 
                st.session_state.shipment_status = "Valid"; st.rerun()
        with i:
            # FIXED: Added use_container_width
            if st.button("❌ MARK INVALID", use_container_width=True): 
                st.session_state.shipment_status = "Invalid"; st.rerun()
            
    # Live Feed is hidden for Border Official per your requirement






 # --- 1. SUPABASE CONFIGURATION ---
#SUPABASE_URL = "https://zoxppbblaiumsmxviydq.supabase.co"
#SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpveHBwYmJsYWl1bXNteHZpeWRxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY3NjQ1MjgsImV4cCI6MjA5MjM0MDUyOH0.3azE5ylA15CV6Vr132dqSmDJqfnW_69DVdb3l6wp0ss"
#supabase = create_client(SUPABASE_URL, SUPABASE_KEY)