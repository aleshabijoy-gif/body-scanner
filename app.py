import os
os.environ['GLIBC_TUNABLES'] = 'glibc.cpu.hwcaps=-XSAVE,-XSAVEC,-AVX2,-GFNI,-AVX'

import streamlit as st
import numpy as np
from PIL import Image
import urllib.request

# Download pose landmarker model
@st.cache_resource
def download_model():
    model_path = "/tmp/pose_landmarker.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite.task"
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None
    return model_path

# Load mediapipe with model
@st.cache_resource
def load_mediapipe():
    try:
        model_path = download_model()
        if not model_path:
            return None
            
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False
        )
        return vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        st.error(f"Failed to load MediaPipe: {e}")
        return None

detector = load_mediapipe()

st.set_page_config(page_title="BodyScan AI", layout="centered")

st.title("🧍 Body Measurement Capture")
st.write("Please enter your height and align with the camera.")

height_cm = st.number_input("Height (cm)", value=170.0)

captured_file = st.camera_input("Take your Front Profile")

if captured_file and detector:
    img = Image.open(captured_file)
    frame = np.array(img)
    
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
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
elif not detector:
    st.error("Please refresh the page to load MediaPipe")
