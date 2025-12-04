import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import cv2
import io
import zipfile
import os

# --------------------------------------------------------------------
#  PAGE CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    page_title="VisionCraft AI",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
#  INITIALIZE SESSION STATE
# --------------------------------------------------------------------
if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"
if 'image_adjustments' not in st.session_state:
    st.session_state.image_adjustments = {
        'brightness': 1.0,
        'contrast': 1.0,
        'saturation': 1.0,
        'sharpness': 1.0,
        'exposure': 1.0,
        'temperature': 1.0,
        'clarity': 1.0
    }
if 'processing_params' not in st.session_state:
    st.session_state.processing_params = {
        'mask_level': 0.5,
        'resolution': '512x512'
    }

# --------------------------------------------------------------------
#  CLEAN CSS - LIKE REMOVE.BG
# --------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    
    .demo-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .demo-title {
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
        color: #2c3e50;
    }
    
    .image-label {
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
#  MODEL LOADING - USING .pth FILE
# --------------------------------------------------------------------
@st.cache_resource
def load_segmentation_model():
    MODEL_PATH = "best_mobilenetv3_unet.pth"
    try:
        model = smp.Unet(
            encoder_name="timm-mobilenetv3_large_100", 
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            st.warning(f"Model file not found: {MODEL_PATH}")
            return None, False
            
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

# Load model
model, MODEL_LOADED = load_segmentation_model()

# --------------------------------------------------------------------
#  UTILITY FUNCTIONS
# --------------------------------------------------------------------
def preprocess_image(image, size=(512, 512)):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def create_mask_overlay(image, mask, alpha=0.6):
    image_np = np.array(image)
    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    
    colored_mask = np.zeros_like(image_np)
    colored_mask[mask_resized > 0.3] = [0, 255, 255]
    
    overlay = cv2.addWeighted(image_np, 1 - alpha, colored_mask, alpha, 0)
    return overlay

def create_black_background_cutout(image, mask):
    """Create cutout with BLACK background"""
    image_np = np.array(image)
    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    
    # Create black background
    black_bg = np.zeros_like(image_np)
    
    # Copy object pixels where mask is > 0
    object_mask = mask_resized > 0.3
    for i in range(3):
        black_bg[:, :, i] = image_np[:, :, i] * object_mask
    
    return black_bg

def apply_image_adjustments(image, adjustments):
    img = image.copy()
    
    if adjustments['brightness'] != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(float(adjustments['brightness']))
    
    if adjustments['contrast'] != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(float(adjustments['contrast']))
    
    if adjustments['saturation'] != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(float(adjustments['saturation']))
    
    if adjustments['sharpness'] != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(float(adjustments['sharpness']))
    
    if adjustments['exposure'] != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(float(adjustments['exposure']))
    
    if adjustments['temperature'] != 1.0:
        img_np = np.array(img, dtype=np.float32)
        temp_val = float(adjustments['temperature'])
        if temp_val > 1.0:
            img_np[:, :, 0] = np.clip(img_np[:, :, 0] * temp_val, 0, 255)
        else:
            img_np[:, :, 2] = np.clip(img_np[:, :, 2] * (2 - temp_val), 0, 255)
        img = Image.fromarray(img_np.astype(np.uint8))
    
    if adjustments['clarity'] != 1.0:
        clarity_val = float(adjustments['clarity'])
        img = img.filter(ImageFilter.UnsharpMask(
            radius=2, 
            percent=int((clarity_val - 1) * 150), 
            threshold=3
        ))
    
    return img

def create_zip_file(images_dict):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for name, image in images_dict.items():
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            zip_file.writestr(f"{name}.png", img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

def reset_adjustments():
    st.session_state.image_adjustments = {
        'brightness': 1.0,
        'contrast': 1.0,
        'saturation': 1.0,
        'sharpness': 1.0,
        'exposure': 1.0,
        'temperature': 1.0,
        'clarity': 1.0
    }
    st.session_state.processing_params['mask_level'] = 0.5

def process_image_optimized(uploaded_file):
    if uploaded_file is None:
        return None
    
    try:
        original_img = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("Processing..."):
            adjusted_img = apply_image_adjustments(original_img, st.session_state.image_adjustments)
            
            size_map = {"256x256": (256, 256), "384x384": (384, 384), "512x512": (512, 512)}
            proc_size = size_map.get(st.session_state.processing_params['resolution'], (512, 512))
            
            x = preprocess_image(adjusted_img, size=proc_size)
            
            with torch.no_grad():
                pred = torch.sigmoid(model(x))[0, 0].numpy()
            
            mask_threshold = st.session_state.processing_params['mask_level']
            mask = (pred > mask_threshold).astype(np.uint8)
            
            original_np = np.array(adjusted_img)
            mask_original_size = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]))
            
            # Create BLACK background cutout
            cutout_black = create_black_background_cutout(adjusted_img, mask)
            
            overlay = create_mask_overlay(adjusted_img, mask)
            
            result = {
                'original': original_img,
                'adjusted': adjusted_img,
                'cutout': cutout_black,  # This is now BLACK background
                'overlay': overlay,
                'mask': mask_original_size
            }
            
            st.session_state.processed_image = result
            st.session_state.original_image = original_img
            
        return result
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# --------------------------------------------------------------------
#  HOME PAGE - CLEAN AND PROFESSIONAL
# --------------------------------------------------------------------
def show_home_page():
    # Header
    st.markdown('<div class="main-header">VisionCraft AI</div>', unsafe_allow_html=True)
    
    # Features - Small and compact
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Lightning Fast</div>
            <div class="feature-desc">Quick image processing with AI technology</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Pixel Precision</div>
            <div class="feature-desc">Accurate object detection and segmentation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔧</div>
            <div class="feature-title">Professional Tools</div>
            <div class="feature-desc">Advanced controls for perfect results</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo Section - Only "See The Transformation"
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.markdown('<div class="demo-title">See The Transformation</div>', unsafe_allow_html=True)
    
    # Image Comparison - Small and horizontal
    col1, col2, col3 = st.columns([1, 0.1, 1])
    
    with col1:
        try:
            sample_before_path = "sample_original2.jpg"
            if os.path.exists(sample_before_path):
                sample_before = Image.open(sample_before_path)
                st.image(sample_before, width=250)
                st.markdown('<div class="image-label">Original</div>', unsafe_allow_html=True)
            else:
                st.image("https://via.placeholder.com/250x200/667eea/ffffff?text=ORIGINAL", width=250)
                st.markdown('<div class="image-label">Original</div>', unsafe_allow_html=True)
        except:
            st.image("https://via.placeholder.com/250x200/667eea/ffffff?text=ORIGINAL", width=250)
            st.markdown('<div class="image-label">Original</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="display: flex; align-items: center; justify-content: center; height: 200px; font-size: 1.5rem; color: #2c3e50;">→</div>', unsafe_allow_html=True)
    
    with col3:
        try:
            sample_after_path = "sample_output2.jpg"
            if os.path.exists(sample_after_path):
                sample_after = Image.open(sample_after_path)
                st.image(sample_after, width=250)
                st.markdown('<div class="image-label">Cutout</div>', unsafe_allow_html=True)
            else:
                st.image("https://via.placeholder.com/250x200/96CEB4/2c3e50?text=CUTOUT", width=250)
                st.markdown('<div class="image-label">Cutout</div>', unsafe_allow_html=True)
        except:
            st.image("https://via.placeholder.com/250x200/96CEB4/2c3e50?text=CUTOUT", width=250)
            st.markdown('<div class="image-label">Cutout</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # CTA Button
    if st.button("TRY IT", use_container_width=True, type="primary"):
        st.session_state.show_uploader = True
        st.session_state.current_page = "process"
        st.rerun()

def show_settings_page():
    st.markdown('<div class="main-header">SETTINGS</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Parameters")
        processing_size = st.selectbox(
            "Processing Resolution",
            options=["256x256", "384x384", "512x512"],
            index=2
        )
        
        mask_level = st.slider(
            "MASK LEVEL",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1
        )
        
        st.session_state.processing_params['resolution'] = processing_size
        st.session_state.processing_params['mask_level'] = mask_level
    
    with col2:
        st.subheader("Performance")
        st.checkbox("Enable GPU Acceleration", value=False)
        st.checkbox("Enable Batch Processing", value=False)
        
        if st.button("Apply Settings", use_container_width=True):
            st.success("Settings applied")

def show_manual_edit_page():
    st.markdown('<div class="main-header">MANUAL EDIT</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_image is None:
        st.warning("Please process an image first")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tab1, tab2 = st.tabs(["CUTOUT", "MASK OVERLAY"])
        
        with tab1:
            cutout_display = st.session_state.processed_image['cutout'][:, :, :3]
            st.image(cutout_display)
        
        with tab2:
            st.image(st.session_state.processed_image['overlay'])
    
    with col2:
        st.subheader("Manual Settings")
        
        new_mask_level = st.slider(
            "MASK LEVEL",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.processing_params['mask_level'],
            step=0.1
        )
        
        new_resolution = st.selectbox(
            "RESOLUTION",
            options=["256x256", "384x384", "512x512"],
            index=2
        )
        
        st.session_state.processing_params['mask_level'] = new_mask_level
        st.session_state.processing_params['resolution'] = new_resolution
        
        st.subheader("Advanced Adjustments")
        
        exposure = st.slider(
            "Exposure", 0.5, 2.0, st.session_state.image_adjustments['exposure'], 0.1
        )
        temperature = st.slider(
            "Temperature", 0.5, 2.0, st.session_state.image_adjustments['temperature'], 0.1
        )
        clarity = st.slider(
            "Clarity", 0.5, 2.0, st.session_state.image_adjustments['clarity'], 0.1
        )
        
        st.session_state.image_adjustments['exposure'] = exposure
        st.session_state.image_adjustments['temperature'] = temperature
        st.session_state.image_adjustments['clarity'] = clarity
        
        if st.button("Reset Settings", use_container_width=True):
            reset_adjustments()
            st.success("Settings reset")

def show_export_page():
    st.markdown('<div class="main-header">EXPORT</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_image is None:
        st.warning("No image to export")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        cutout_display = st.session_state.processed_image['cutout'][:, :, :3]
        st.image(cutout_display)
        st.caption("CUTOUT")
    
    with col2:
        st.image(st.session_state.processed_image['overlay'])
        st.caption("MASK OVERLAY")
    
    st.subheader("Download")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        buf_png = io.BytesIO()
        cutout_pil = Image.fromarray(st.session_state.processed_image['cutout'])
        cutout_pil.save(buf_png, format="PNG")
        st.download_button(
            "PNG FORMAT",
            data=buf_png.getvalue(),
            file_name="cutout.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col2:
        buf_jpg = io.BytesIO()
        cutout_rgb = Image.fromarray(st.session_state.processed_image['cutout'][:, :, :3])
        cutout_rgb.save(buf_jpg, format="JPEG", quality=95)
        st.download_button(
            "JPG FORMAT",
            data=buf_jpg.getvalue(),
            file_name="cutout.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    
    with col3:
        buf_hq = io.BytesIO()
        cutout_pil.save(buf_hq, format="PNG", optimize=True)
        st.download_button(
            "HIGH QUALITY",
            data=buf_hq.getvalue(),
            file_name="high_quality.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col4:
        zip_data = create_zip_file({
            "cutout": Image.fromarray(st.session_state.processed_image['cutout']),
            "overlay": Image.fromarray(st.session_state.processed_image['overlay'])
        })
        st.download_button(
            "ZIPFILE",
            data=zip_data.getvalue(),
            file_name="results.zip",
            mime="application/zip",
            use_container_width=True
        )

def render_sidebar():
    with st.sidebar:
        st.header("Image Adjustments")
        
        st.subheader("Basic Adjustments")
        st.session_state.image_adjustments['brightness'] = st.slider(
            "Brightness", 0.5, 2.0, st.session_state.image_adjustments['brightness'], 0.1
        )
        st.session_state.image_adjustments['contrast'] = st.slider(
            "Contrast", 0.5, 2.0, st.session_state.image_adjustments['contrast'], 0.1
        )
        st.session_state.image_adjustments['saturation'] = st.slider(
            "Saturation", 0.5, 2.0, st.session_state.image_adjustments['saturation'], 0.1
        )
        st.session_state.image_adjustments['sharpness'] = st.slider(
            "Sharpness", 0.5, 2.0, st.session_state.image_adjustments['sharpness'], 0.1
        )
        
        st.subheader("Advanced Adjustments")
        st.session_state.image_adjustments['exposure'] = st.slider(
            "Exposure", 0.5, 2.0, st.session_state.image_adjustments['exposure'], 0.1
        )
        st.session_state.image_adjustments['temperature'] = st.slider(
            "Temperature", 0.5, 2.0, st.session_state.image_adjustments['temperature'], 0.1
        )
        st.session_state.image_adjustments['clarity'] = st.slider(
            "Clarity", 0.5, 2.0, st.session_state.image_adjustments['clarity'], 0.1
        )
        
        if st.button("Reset Adjustments", use_container_width=True):
            reset_adjustments()
            st.success("Adjustments reset")
        
        st.markdown("---")
        st.subheader("Processing Settings")
        st.session_state.processing_params['resolution'] = st.selectbox(
            "Resolution",
            options=["256x256", "384x384", "512x512"],
            index=2
        )
        st.session_state.processing_params['mask_level'] = st.slider(
            "MASK LEVEL",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.processing_params['mask_level'],
            step=0.1
        )

def main():
    # Navigation
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("Home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
    
    with nav_col2:
        if st.button("Settings", use_container_width=True):
            st.session_state.current_page = "settings"
            st.rerun()
    
    with nav_col3:
        if st.button("Manual Edit", use_container_width=True):
            st.session_state.current_page = "manual"
            st.rerun()
    
    with nav_col4:
        if st.button("Export", use_container_width=True):
            st.session_state.current_page = "export"
            st.rerun()
    
    # Show current page
    if st.session_state.current_page == "home":
        show_home_page()
    elif st.session_state.current_page == "settings":
        show_settings_page()
    elif st.session_state.current_page == "manual":
        show_manual_edit_page()
    elif st.session_state.current_page == "export":
        show_export_page()
    
    # Processing Section
    if st.session_state.show_uploader:
        st.markdown("---")
        st.markdown('<div class="main-header">PROCESSING</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload your image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None and MODEL_LOADED:
            render_sidebar()
            processed_data = process_image_optimized(uploaded_file)
            
            if processed_data:
                st.success("Image processed successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(processed_data['original'])
                    st.caption("Original")
                    st.image(processed_data['adjusted'])
                    st.caption("Adjusted")
                
                with col2:
                    cutout_display = processed_data['cutout'][:, :, :3]
                    st.image(cutout_display)
                    st.caption("CUTOUT")
                    st.image(processed_data['overlay'])
                    st.caption("MASK OVERLAY")
                
                if st.button("Process New Image", use_container_width=True):
                    st.session_state.show_uploader = False
                    st.session_state.processed_image = None
                    st.session_state.original_image = None
                    st.session_state.current_page = "home"
                    st.rerun()

if __name__ == "__main__":
    main()
