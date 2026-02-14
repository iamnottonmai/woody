import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
from PIL import Image
import gdown
import os
from ultralytics import YOLO
import av

# Page config
st.set_page_config(
    page_title="Woody - Wood Type Classifier",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Lora:wght@400;500;600&display=swap');
    
    /* Main theme colors */
    :root {
        --wood-brown: #8B4513;
        --wood-dark: #3E2723;
        --wood-light: #D7CCC8;
        --accent-gold: #FFD700;
        --bg-cream: #FFF8DC;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #FFF8DC 0%, #F5E6D3 100%);
    }
    
    /* Headers */
    h1 {
        font-family: 'Playfair Display', serif;
        color: #3E2723;
        font-weight: 900;
        font-size: 4.5rem !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: 2px;
    }
    
    h2 {
        font-family: 'Playfair Display', serif;
        color: #8B4513;
        font-weight: 700;
        border-bottom: 3px solid #FFD700;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        font-family: 'Lora', serif;
        color: #3E2723;
        font-weight: 600;
    }
    
    /* Subtitle */
    .subtitle {
        font-family: 'Lora', serif;
        font-size: 1.3rem;
        color: #8B4513;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Card containers */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        border: 2px solid #D7CCC8;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8B4513 0%, #654321 100%);
        color: white;
        font-family: 'Lora', serif;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(139, 69, 19, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(139, 69, 19, 0.4);
        background: linear-gradient(135deg, #654321 0%, #8B4513 100%);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #8B4513;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #3E2723 0%, #5D4037 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3E2723 0%, #5D4037 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #FFF8DC !important;
    }
    
    /* Detection results */
    .detection-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #8B4513;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .confidence-bar {
        background: #D7CCC8;
        border-radius: 10px;
        height: 25px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #8B4513 0%, #FFD700 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-family: 'Lora', serif;
        transition: width 0.5s ease;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FFD700;
        margin: 1rem 0;
        font-family: 'Lora', serif;
    }
    
    /* Stats display */
    .stat-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 2rem 0;
    }
    
    .stat-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        min-width: 150px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-top: 4px solid #8B4513;
        margin: 0.5rem;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        color: #8B4513;
        font-family: 'Playfair Display', serif;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        font-family: 'Lora', serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Model loading function
@st.cache_resource
def load_model():
    """Download and load the YOLO model"""
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        with st.spinner("🌲 Downloading Woody model... Please wait"):
            try:
                # Extract file ID from the Google Drive link
                file_id = "1thy-dO8ugVAUpfdrepi662dq3QOvIkMg"
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Class names for wood types
CLASS_NAMES = [
    "Chair",
    "Liquor stand",
    "chair",
    "chair 3",
    "chair1",
    "chair2",
    "coaster",
    "counter",
    "non",
    "table",
    "wood",
    "woodCoaster",
    "table 2"
]

def process_image(image, model, confidence_threshold=0.25):
    """Process image with YOLO model"""
    # Run inference
    results = model(image, conf=confidence_threshold)
    
    # Get the first result
    result = results[0]
    
    # Draw bounding boxes on image
    annotated_image = result.plot()
    
    # Extract detection details
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            
            detections.append({
                'class': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}",
                'confidence': confidence,
                'bbox': bbox
            })
    
    return annotated_image, detections

def main():
    # Header
    st.markdown("<h1>🪵 WOODY</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Intelligent Wood Type Classification System</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        st.markdown("---")
        st.markdown("## 📋 Wood Types")
        st.markdown("Woody can identify:")
        for i, class_name in enumerate(CLASS_NAMES, 1):
            st.markdown(f"• {class_name}")
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.markdown("""
        **Woody** uses YOLOv8 object detection to classify different wood types and furniture from images.
        
        Simply upload an image or use your camera to get started!
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check your internet connection and try again.")
        return
    
    # Main content
    st.markdown("## 📸 Choose Input Method")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera Input"])
    
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of wood or furniture"
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("### Detection Results")
                with st.spinner("🔍 Analyzing image..."):
                    annotated_image, detections = process_image(
                        image_np, 
                        model, 
                        confidence_threshold
                    )
                    st.image(annotated_image, use_container_width=True)
            
            # Display detections
            if detections:
                st.markdown("## 🎯 Detected Objects")
                
                # Statistics
                st.markdown("<div class='stat-container'>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-number'>{len(detections)}</div>
                    <div class='stat-label'>Objects Found</div>
                </div>
                """, unsafe_allow_html=True)
                
                if detections:
                    avg_conf = sum(d['confidence'] for d in detections) / len(detections)
                    st.markdown(f"""
                    <div class='stat-box'>
                        <div class='stat-number'>{avg_conf*100:.1f}%</div>
                        <div class='stat-label'>Avg Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Individual detections
                for i, det in enumerate(detections, 1):
                    st.markdown(f"""
                    <div class='detection-box'>
                        <h3 style='margin-top: 0;'>Detection #{i}: {det['class']}</h3>
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width: {det['confidence']*100}%;'>
                                {det['confidence']*100:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='info-box'>
                    ℹ️ No objects detected. Try adjusting the confidence threshold or upload a different image.
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            📷 Click the button below to capture an image using your camera.
        </div>
        """, unsafe_allow_html=True)
        
        camera_photo = st.camera_input("Take a picture")
        
        if camera_photo is not None:
            # Read image from camera
            image = Image.open(camera_photo)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Captured Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("### Detection Results")
                with st.spinner("🔍 Analyzing image..."):
                    annotated_image, detections = process_image(
                        image_np, 
                        model, 
                        confidence_threshold
                    )
                    st.image(annotated_image, use_container_width=True)
            
            # Display detections
            if detections:
                st.markdown("## 🎯 Detected Objects")
                
                # Statistics
                st.markdown("<div class='stat-container'>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-number'>{len(detections)}</div>
                    <div class='stat-label'>Objects Found</div>
                </div>
                """, unsafe_allow_html=True)
                
                if detections:
                    avg_conf = sum(d['confidence'] for d in detections) / len(detections)
                    st.markdown(f"""
                    <div class='stat-box'>
                        <div class='stat-number'>{avg_conf*100:.1f}%</div>
                        <div class='stat-label'>Avg Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Individual detections
                for i, det in enumerate(detections, 1):
                    st.markdown(f"""
                    <div class='detection-box'>
                        <h3 style='margin-top: 0;'>Detection #{i}: {det['class']}</h3>
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width: {det['confidence']*100}%;'>
                                {det['confidence']*100:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='info-box'>
                    ℹ️ No objects detected. Try adjusting the confidence threshold or take a different picture.
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
