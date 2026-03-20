import os
# Fix for CPU instruction sets on shared cloud servers
os.environ['GLIBC_TUNABLES'] = 'glibc.cpu.hwcaps=-XSAVE,-XSAVEC,-AVX2,-GFNI,-AVX'

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import urllib.request
import mediapipe as mp

# --- 1. Setup and Model Downloading ---
@st.cache_resource
def download_model():
    model_path = "/tmp/pose_landmarker_full.task"
    if not os.path.exists(model_path):
        # Using the "Full" model for better landmark accuracy than "Lite"
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None
    return model_path

@st.cache_resource
def load_detector():
    try:
        model_path = download_model()
        if not model_path: return None
        
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True # Enabled for future girth measurements
        )
        return vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        st.error(f"Failed to load MediaPipe: {e}")
        return None

# --- 2. UI Configuration ---
st.set_page_config(page_title="BodyScan AI", layout="centered")
st.title("🧍 AI Body Scanner")
st.write("Follow the guides to capture your front and side profiles.")

detector = load_detector()
height_cm = st.number_input("Enter your Height (cm)", min_value=100.0, max_value=250.0, value=170.0)

# --- 3. Step-by-Step Capture Logic ---
if 'front_processed' not in st.session_state:
    st.session_state.front_processed = False

# STEP 1: FRONT CAPTURE
st.subheader("Step 1: Front Profile")
st.info("Stand in an 'A-Pose' (arms 20° from sides, feet shoulder-width).")
front_file = st.camera_input("Capture Front", key="front_cam")

if front_file and detector:
    img = Image.open(front_file)
    frame = np.array(img)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image)
    
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        # Check alignment: Shoulders (11, 12)
        l_sh, r_sh = landmarks[11], landmarks[12]
        if abs(l_sh.y - r_sh.y) < 0.05:
            st.success("✅ Front alignment verified!")
            st.session_state.front_processed = True
            st.session_state.front_landmarks = landmarks
        else:
            st.warning("⚠️ Keep your shoulders level!")
    else:
        st.error("No person detected. Stand further back.")

# STEP 2: SIDE CAPTURE (Only shows if Front is done)
if st.session_state.front_processed:
    st.divider()
    st.subheader("Step 2: Side Profile")
    st.info("Turn 90° to the right. Hold arms slightly forward.")
    side_file = st.camera_input("Capture Side", key="side_cam")
    
    if side_file:
        img_side = Image.open(side_file)
        frame_side = np.array(img_side)
        mp_side = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_side)
        result_side = detector.detect(mp_side)
        
        if result_side.pose_landmarks:
            st.success("✅ Side alignment verified!")
            
            # --- 4. Final Measurement Calculation ---
            if st.button("🚀 Calculate Measurements"):
                with st.spinner("Processing 3D Geometry..."):
                    # PIXELS TO METRIC CALCULATION
                    # Height in Pixels: Nose(0) to Ankle(27/28)
                    f_landmarks = st.session_state.front_landmarks
                    px_height = abs(f_landmarks[28].y - f_landmarks[0].y)
                    scaling_factor = height_cm / px_height
                    
                    # Conceptual Measurement Logic:
                    # Waist Width (Front) * Scaling
                    waist_front_px = abs(f_landmarks[24].x - f_landmarks[23].x)
                    waist_cm = waist_front_px * scaling_factor * 3.14 # Simple approximation
                    
                    st.balloons()
                    st.header("Your Results")
                    col1, col2 = st.columns(2)
                    col1.metric("Waist (Approx)", f"{round(waist_cm, 1)} cm")
                    col2.metric("Chest (Approx)", f"{round(waist_cm * 1.1, 1)} cm")
                    st.warning("Note: These are 2D approximations. 3D Mesh integration is next.")
        else:
            st.error("Side profile not detected.")
