import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize pose landmarker
base_options = python.BaseOptions(model_asset_path=None)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

st.set_page_config(page_title="BodyScan AI", layout="centered")

st.title("🧍 Body Measurement Capture")
st.write("Please enter your height and align with the camera.")

height_cm = st.number_input("Height (cm)", value=170.0)

captured_file = st.camera_input("Take your Front Profile")

if captured_file:
    img = Image.open(captured_file)
    frame = np.array(img)
    
    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    # Detect pose landmarks
    detection_result = detector.detect(mp_image)

    if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
        landmarks = detection_result.pose_landmarks[0]
        
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        
        l_sh = landmarks[LEFT_SHOULDER]
        r_sh = landmarks[RIGHT_SHOULDER]
        
        if abs(l_sh.y - r_sh.y) < 0.05:
            st.success("✅ Alignment looks good!")
            if 'front_img' not in st.session_state:
                st.session_state['front_img'] = frame
        else:
            st.warning("⚠️ Please stand straight! Your shoulders are tilted.")
    else:
        st.error("No person detected. Try standing further back.")
