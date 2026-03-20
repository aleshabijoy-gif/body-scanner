import streamlit as st
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except:
    st.error("MediaPipe failed to load")
    st.stop()

# Initialize pose detector
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

st.set_page_config(page_title="BodyScan AI", layout="centered")

st.title("🧍 Body Measurement Capture")
st.write("Please enter your height and align with the camera.")

height_cm = st.number_input("Height (cm)", value=170.0)

captured_file = st.camera_input("Take your Front Profile")

if captured_file:
    img = Image.open(captured_file)
    frame = np.array(img)
    
    # Convert RGB to BGR for MediaPipe
    import cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = pose.process(frame_bgr)

    if results.pose_landmarks:
        l_sh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        if abs(l_sh.y - r_sh.y) < 0.05:
            st.success("✅ Alignment looks good!")
            if 'front_img' not in st.session_state:
                st.session_state['front_img'] = frame
        else:
            st.warning("⚠️ Please stand straight! Your shoulders are tilted.")
    else:
        st.error("No person detected. Try standing further back.")
