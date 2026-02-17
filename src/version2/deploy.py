import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

from model import TransformerOCR
from utils import KhmerLabelConverter
from datasets import load_dataset

# Set page config
st.set_page_config(
    page_title="Khmer OCR",
    page_icon="📝",
    layout="centered"
)
@st.cache_resource
def load_model_and_converter():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            "checkpoints",
            "sprint2_model.pth"
        )

        if not os.path.exists(checkpoint_path):
            st.error(f"❌ Model checkpoint not found at: {checkpoint_path}")
            return None, None, None

        # Load checkpoint FIRST
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # 🔥 Get num_classes from trained model
        num_classes = checkpoint["fc.weight"].shape[0]
        st.write(f"Trained model vocab size: {num_classes}")

        # 🔥 IMPORTANT: Use FULL training dataset (not [:2000])
        raw_data = load_dataset("seanghay/khmer-hanuman-100k", split="train")
        all_text = "".join([x['text'] for x in raw_data])
        vocab = sorted(list(set(all_text)))

        st.write(f"Rebuilt vocab size: {len(vocab)}")

        if len(vocab) != num_classes:
            st.error("⚠️ Vocab size mismatch with trained model!")
            return None, None, None

        converter = KhmerLabelConverter(vocab)

        model = TransformerOCR(
            num_classes=num_classes
        ).to(device)

        model.load_state_dict(checkpoint)
        model.eval()

        st.success("✅ Model loaded successfully!")

        return model, converter, device

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


def preprocess_image(image):
    """Preprocess image for the model"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to height 32, maintain aspect ratio
    aspect_ratio = image.width / image.height
    new_width = int(32 * aspect_ratio)
    image = image.resize((new_width, 32), Image.LANCZOS)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict(model, image_tensor, converter, device):
    """Run inference on the image"""
    try:
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            # Forward pass
            preds = model(image_tensor)  # (Time, Batch, NumClasses)
            
            st.write(f"**Debug Info:**")
            st.write(f"- Prediction shape: {preds.shape}")
            st.write(f"- Time steps: {preds.shape[0]}")
            st.write(f"- Num classes: {preds.shape[2]}")
            
            # Greedy decode
            _, max_index = torch.max(preds, dim=2)  # (Time, Batch)
            pred_indices = max_index[:, 0].cpu().numpy().tolist()
            
            st.write(f"- Raw indices (first 20): {pred_indices[:20]}")
            
            # Decode to text
            decoded_text = converter.decode(pred_indices)
            
            st.write(f"- Decoded length: {len(decoded_text)}")
            
        return decoded_text
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return ""

def main():
    st.title("📝 Khmer OCR - Text Recognition")
    st.write("Upload an image with Khmer text to extract the text")
    
    # Load model
    with st.spinner("Loading model..."):
        model, converter, device = load_model_and_converter()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses a Transformer-based OCR model "
        "trained on Khmer text to recognize and extract "
        "text from images."
    )
    
    st.sidebar.header("Model Info")
    st.sidebar.write(f"Device: {device}")
    st.sidebar.write(f"Vocab Size: {converter.get_num_classes()}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload an image containing Khmer text"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            st.write(f"Size: {image.size}")
        
        with col2:
            st.subheader("Preprocessed Image")
            # Preprocess
            preprocessed = preprocess_image(image)
            
            st.write(f"Tensor shape: {preprocessed.shape}")
            
            # Show preprocessed image
            preprocessed_img = preprocessed.squeeze().numpy()
            preprocessed_img = (preprocessed_img * 0.5 + 0.5)  # Denormalize
            st.image(preprocessed_img, use_container_width=True, clamp=True)
        
        # Predict button
        if st.button("🔍 Recognize Text", type="primary"):
            with st.spinner("Processing..."):
                # Run prediction
                result = predict(model, preprocessed, converter, device)
                
                # Display result
                st.markdown("---")
                st.success("Recognition Complete!")
                
                if result and len(result.strip()) > 0:
                    st.subheader("Extracted Text:")
                    st.markdown(f"### {result}")
                    
                    # Copy button
                    st.text_area("Copy text from here:", result, height=100)
                else:
                    st.warning("⚠️ No text detected or empty prediction")
                    st.info("This could mean:\n- The model needs more training\n- The image quality is poor\n- The text is not similar to training data")
    
    # Example section
    st.markdown("---")
    st.subheader("How to use:")
    st.markdown("""
    1. Upload an image containing Khmer text
    2. Click the **Recognize Text** button
    3. View the extracted text below
    
    **Tips:**
    - Use clear, high-contrast images for best results
    - Single-line text works best
    - Ensure the text is horizontal and not rotated
    """)

if __name__ == "__main__":
    main()