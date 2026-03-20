import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

st.set_page_config(page_title="BodyScan AI", layout="centered")

st.title("🧍 Body Measurement Capture")
st.write("Please enter your height and align with the camera.")

height_cm = st.number_input("Height (cm)", value=170.0)

# Real-time Alignment Logic
captured_file = st.camera_input("Take your Front Profile")

if captured_file:
    # Process Image
    img = Image.open(captured_file)
    frame = np.array(img)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if results.pose_landmarks:
        # Simple alignment check: Are shoulders level?
        l_sh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        if abs(l_sh.y - r_sh.y) < 0.05:
            st.success("✅ Alignment looks good!")
            st.session_state['front_img'] = frame
        else:
            st.warning("⚠️ Please stand straight! Your shoulders are tilted.")
    else:
        st.error("No person detected. Try standing further back.")