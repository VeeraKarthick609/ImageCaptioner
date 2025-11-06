import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import CaptioningTeacher
from data_loader import Vocabulary

@st.cache_resource
def load_model_and_vocab():
    """Load the trained teacher model and vocabulary"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Check if model file exists
        model_path = 'saved_models/best_teacher_model.pth'
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None, None
            
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        vocab_size = checkpoint['vocab_size']
        
        st.info(f"📊 Model info: Vocab size = {vocab_size}, Device = {device}")
        
        # Initialize model
        model = CaptioningTeacher(
            vocab_size=vocab_size,
            embed_size=512,
            num_heads=8,
            num_decoder_layers=4,
            dropout=0.15
        ).to(device)
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load vocabulary from dataset
        from data_loader import FlickrDataset
        
        # Check if data files exist
        if not os.path.exists("data/flickr8k/captions_clean.csv"):
            st.error("Captions file not found: data/flickr8k/captions_clean.csv")
            return None, None, None
            
        if not os.path.exists("data/flickr8k/Images"):
            st.error("Images directory not found: data/flickr8k/Images")
            return None, None, None
        
        dataset = FlickrDataset(
            root_dir="data/flickr8k/Images",
            captions_file="data/flickr8k/captions_clean.csv",
            transform=None
        )
        vocab = dataset.vocab
        
        # Validate vocabulary
        required_tokens = ['<START>', '<END>', '<UNK>', '<PAD>']
        missing_tokens = [token for token in required_tokens if token not in vocab.stoi]
        if missing_tokens:
            st.error(f"Missing vocabulary tokens: {missing_tokens}")
            return None, None, None
        
        st.success(f"✅ Vocabulary loaded: {len(vocab)} tokens")
        return model, vocab, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

def preprocess_image(image):
    """Preprocess uploaded image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image)

def generate_caption(model, image_tensor, vocab, device, max_length=25):
    """Generate caption for the input image using the model's built-in method"""
    model.eval()
    
    try:
        # Validate inputs
        if image_tensor is None:
            raise ValueError("Image tensor is None")
        if vocab is None:
            raise ValueError("Vocabulary is None")
            
        # Move image to device
        image_tensor = image_tensor.to(device)
        
        # Validate tensor shape
        if image_tensor.dim() not in [3, 4]:
            raise ValueError(f"Invalid image tensor shape: {image_tensor.shape}")
        
        # Use the model's built-in caption_image method
        with torch.no_grad():
            caption_tokens = model.caption_image(image_tensor, vocab, max_length=max_length)
        
        # Validate output
        if not caption_tokens:
            return "[No caption generated]"
        
        # Join tokens into a string, filtering out any None or empty tokens
        valid_tokens = [token for token in caption_tokens if token and token.strip()]
        
        if not valid_tokens:
            return "[Empty caption generated]"
            
        caption = ' '.join(valid_tokens)
        
        # Final validation
        if not caption.strip():
            return "[Invalid caption generated]"
            
        return caption
        
    except Exception as e:
        st.error(f"Error in caption generation: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return f"[Error: {str(e)}]"

def main():
    st.set_page_config(
        page_title="Image Caption Generator",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("📸 Image Caption Generator")
    st.markdown("Upload an image and let the AI generate a descriptive caption!")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, vocab, device = load_model_and_vocab()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model files exist.")
        st.info("Make sure you have:")
        st.code("""
        - saved_models/best_teacher_model.pth
        - data/flickr8k/captions_clean.csv
        - data/flickr8k/Images/ directory
        """)
        return
    
    st.success(f"✅ Model loaded successfully! Running on: {device}")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image file (JPG, PNG, etc.)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"📊 Image size: {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        st.header("Generated Caption")
        
        if uploaded_file is not None:
            # Generate caption button
            if st.button("🔮 Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
                    try:
                        # Show processing info
                        st.info(f"🔄 Processing image: {image.size[0]}x{image.size[1]} pixels")
                        
                        # Preprocess image
                        image_tensor = preprocess_image(image)
                        st.info(f"✅ Image preprocessed: {image_tensor.shape}")
                        
                        # Generate caption
                        caption = generate_caption(model, image_tensor, vocab, device)
                        
                        if caption.strip() and not caption.startswith('['):
                            # Display result
                            st.success("Caption generated successfully!")
                            st.markdown(f"### 💬 Caption:")
                            st.markdown(f"*\"{caption}\"*")
                            
                            # Additional info
                            word_count = len(caption.split())
                            st.info(f"📝 Word count: {word_count}")
                            
                            # Show some debug info
                            with st.expander("🔍 Debug Information"):
                                st.write(f"**Device:** {device}")
                                st.write(f"**Image tensor shape:** {image_tensor.shape}")
                                st.write(f"**Generated tokens:** {caption.split()}")
                            
                        else:
                            st.warning(f"Caption generation issue: {caption}")
                            st.info("Try with a different image or check the debug information.")
                            
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.info("👆 Upload an image to generate a caption")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("🤖 Model Information")
        st.markdown("""
        **Teacher Model Architecture:**
        - Vision Transformer (ViT-Small) Encoder
        - Transformer Decoder (4 layers)
        - 8 attention heads
        - 512 embedding dimensions
        
        **Performance Metrics:**
        - BLEU-1: 58.5%
        - METEOR: 55.1%
        - Success Rate: 100%
        - Avg Caption Length: 11.3 words
        """)
        
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload an image using the file uploader
        2. Click "Generate Caption" button
        3. Wait for the AI to process your image
        4. View the generated caption!
        
        **Supported formats:** JPG, PNG, BMP, TIFF
        """)
        
        st.header("⚡ Tips")
        st.markdown("""
        - Clear, well-lit images work best
        - The model was trained on Flickr8k dataset
        - Common objects and scenes get better captions
        - Processing time depends on your hardware
        """)

if __name__ == "__main__":
    main()
