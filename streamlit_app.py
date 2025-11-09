import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from teacher_model import CaptioningTeacher
from student_model import CaptioningStudent
from data_loader import Vocabulary

@st.cache_resource
def load_models_and_vocab():
    """Load both teacher and student models with vocabulary"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load vocabulary from dataset first
        from data_loader import FlickrDataset
        
        # Check if data files exist
        if not os.path.exists("data/flickr8k/captions_clean.csv"):
            st.error("Captions file not found: data/flickr8k/captions_clean.csv")
            return None, None, None, None
            
        if not os.path.exists("data/flickr8k/Images"):
            st.error("Images directory not found: data/flickr8k/Images")
            return None, None, None, None
        
        dataset = FlickrDataset(
            root_dir="data/flickr8k/Images",
            captions_file="data/flickr8k/captions_clean.csv",
            transform=None
        )
        vocab = dataset.vocab
        vocab_size = len(vocab)
        
        # Validate vocabulary
        required_tokens = ['<START>', '<END>', '<UNK>', '<PAD>']
        missing_tokens = [token for token in required_tokens if token not in vocab.stoi]
        if missing_tokens:
            st.error(f"Missing vocabulary tokens: {missing_tokens}")
            return None, None, None, None
        
        st.info(f"📊 Vocab size: {vocab_size}, Device: {device}")
        
        # Load Teacher Model
        teacher_model = None
        teacher_path = 'saved_models/best_teacher_model.pth'
        if os.path.exists(teacher_path):
            teacher_checkpoint = torch.load(teacher_path, map_location=device)
            teacher_model = CaptioningTeacher(
                vocab_size=vocab_size,
                embed_size=512,
                num_heads=8,
                num_decoder_layers=4,
                dropout=0.15
            ).to(device)
            teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
            teacher_model.eval()
            st.success("✅ Teacher model loaded")
        else:
            st.warning(f"Teacher model not found: {teacher_path}")
        
        # Load Student Model
        student_model = None
        student_path = 'saved_models/best_student_model.pth'
        if os.path.exists(student_path):
            student_checkpoint = torch.load(student_path, map_location=device, weights_only=False)
            student_model = CaptioningStudent(
                vocab_size=vocab_size,
                embed_size=256,
                hidden_size=512,
                num_layers=2,
                dropout=0.2
            ).to(device)
            student_model.load_state_dict(student_checkpoint["student_state_dict"])
            student_model.eval()
            st.success("✅ Student model loaded")
        else:
            st.warning(f"Student model not found: {student_path}")
        
        if teacher_model is None and student_model is None:
            st.error("No models found! Please train at least one model first.")
            return None, None, None, None
        
        st.success(f"✅ Vocabulary loaded: {vocab_size} tokens")
        return teacher_model, student_model, vocab, device
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None

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

def generate_caption(model, image_tensor, vocab, device, model_name="Model", max_length=25):
    """Generate caption for the input image using the model's built-in method"""
    if model is None:
        return f"[{model_name} not available]"
    
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
            caption_result = model.caption_image(image_tensor, vocab, max_length=max_length)
        
        # Handle different return types (teacher returns list, student might return string)
        if isinstance(caption_result, list):
            if not caption_result:
                return "[No caption generated]"
            caption_tokens = caption_result[0] if caption_result else ""
        else:
            caption_tokens = caption_result
        
        # Validate output
        if not caption_tokens:
            return "[No caption generated]"
        
        # Handle string vs token list
        if isinstance(caption_tokens, str):
            caption = caption_tokens
        else:
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
        return f"[Error in {model_name}: {str(e)}]"

def main():
    st.set_page_config(
        page_title="Image Caption Generator - Teacher vs Student",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("📸 Image Caption Generator - Teacher vs Student Models")
    st.markdown("Upload an image and compare captions from both the Teacher (ViT-Transformer) and Student (CNN-LSTM) models!")
    
    # Load models
    with st.spinner("Loading AI models..."):
        teacher_model, student_model, vocab, device = load_models_and_vocab()
    
    if teacher_model is None and student_model is None:
        st.error("Failed to load any models. Please check if the model files exist.")
        st.info("Make sure you have at least one of:")
        st.code("""
        - saved_models/best_teacher_model.pth (Teacher Model)
        - saved_models/best_student_model.pth (Student Model)
        - data/flickr8k/captions_clean.csv
        - data/flickr8k/Images/ directory
        """)
        return
    
    st.success(f"✅ Models loaded successfully! Running on: {device}")
    
    # Create three columns: image upload, teacher caption, student caption
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.header("📷 Upload Image")
        
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
            st.info(f"📊 Size: {image.size[0]} x {image.size[1]} px")
            
            # Generate caption button
            if st.button("🔮 Generate Captions", type="primary", use_container_width=True):
                # Store in session state to trigger generation
                st.session_state.generate_captions = True
                st.rerun()
    
    with col2:
        st.header("🎓 Teacher Model")
        st.markdown("**ViT-Transformer** (25M params)")
        
        if uploaded_file is not None and st.session_state.get('generate_captions', False):
            with st.spinner("Teacher generating..."):
                try:
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Generate teacher caption
                    teacher_caption = generate_caption(
                        teacher_model, image_tensor, vocab, device, 
                        model_name="Teacher", max_length=25
                    )
                    
                    if teacher_caption.strip() and not teacher_caption.startswith('['):
                        st.success("✅ Caption Generated!")
                        st.markdown("### 💬 Caption:")
                        st.markdown(f"*\"{teacher_caption}\"*")
                        
                        # Stats
                        word_count = len(teacher_caption.split())
                        st.info(f"📝 Words: {word_count}")
                        
                        # Performance info
                        with st.expander("📊 Model Info"):
                            st.write("**Architecture:** Vision Transformer")
                            st.write("**Parameters:** ~25M")
                            st.write("**Encoder:** ViT-Small")
                            st.write("**Decoder:** 4-layer Transformer")
                        
                    else:
                        st.warning(teacher_caption)
                        
                except Exception as e:
                    st.error(f"Teacher error: {str(e)}")
        else:
            if teacher_model is None:
                st.warning("Teacher model not available")
                st.info("Train the teacher model first")
            else:
                st.info("👈 Upload image and click generate")
    
    with col3:
        st.header("🎒 Student Model")
        st.markdown("**CNN-LSTM** (8M params)")
        
        if uploaded_file is not None and st.session_state.get('generate_captions', False):
            with st.spinner("Student generating..."):
                try:
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Generate student caption
                    student_caption = generate_caption(
                        student_model, image_tensor, vocab, device, 
                        model_name="Student", max_length=25
                    )
                    
                    if student_caption.strip() and not student_caption.startswith('['):
                        st.success("✅ Caption Generated!")
                        st.markdown("### 💬 Caption:")
                        st.markdown(f"*\"{student_caption}\"*")
                        
                        # Stats
                        word_count = len(student_caption.split())
                        st.info(f"📝 Words: {word_count}")
                        
                        # Performance info
                        with st.expander("📊 Model Info"):
                            st.write("**Architecture:** CNN-LSTM")
                            st.write("**Parameters:** ~8M (3x smaller)")
                            st.write("**Encoder:** ResNet-50")
                            st.write("**Decoder:** 2-layer LSTM")
                            st.write("**Speed:** ~2-3x faster")
                        
                    else:
                        st.warning(student_caption)
                        
                except Exception as e:
                    st.error(f"Student error: {str(e)}")
        else:
            if student_model is None:
                st.warning("Student model not available")
                st.info("Train the student model first")
            else:
                st.info("👈 Upload image and click generate")
    
    # Reset generation flag
    if st.session_state.get('generate_captions', False):
        st.session_state.generate_captions = False
    
    # Comparison section
    if uploaded_file is not None:
        st.markdown("---")
        st.header("🔍 Model Comparison")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("### 🎓 Teacher Model")
            st.markdown("""
            **Advantages:**
            - Higher accuracy (BLEU: ~58.5%)
            - Better understanding of complex scenes
            - More detailed descriptions
            - State-of-the-art architecture
            
            **Disadvantages:**
            - Larger model size (25M params)
            - Slower inference
            - Higher memory usage
            """)
        
        with comp_col2:
            st.markdown("### 🎒 Student Model")
            st.markdown("""
            **Advantages:**
            - 3x smaller model size (8M params)
            - 2-3x faster inference
            - Lower memory usage
            - Good for deployment
            
            **Target Performance:**
            - 85-95% of teacher accuracy
            - Suitable for mobile/edge devices
            - Real-time applications
            """)
    
    # Clear uploaded file button
    if uploaded_file is not None:
        if st.button("🗑️ Clear Image"):
            st.rerun()
    
    # Sidebar with model info
    with st.sidebar:
        st.header("🤖 Model Information")
        
        # Teacher Model Info
        if teacher_model is not None:
            st.markdown("### 🎓 Teacher Model")
            st.markdown("""
            **Architecture:** ViT-Transformer
            - Vision Transformer (ViT-Small) Encoder
            - Transformer Decoder (4 layers)
            - 8 attention heads, 512 embedding dims
            - **Parameters:** ~25M
            
            **Performance:**
            - BLEU-1: ~58.5%
            - METEOR: ~55.1%
            - High accuracy, detailed captions
            """)
        
        # Student Model Info
        if student_model is not None:
            st.markdown("### 🎒 Student Model")
            st.markdown("""
            **Architecture:** CNN-LSTM
            - ResNet-50 CNN Encoder
            - 2-layer LSTM Decoder
            - Attention mechanism
            - **Parameters:** ~8M (3x smaller)
            
            **Performance:**
            - Target: 85-95% of teacher
            - 2-3x faster inference
            - Optimized for deployment
            """)
        
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload an image using the file uploader
        2. Click "Generate Captions" button
        3. Compare outputs from both models
        4. Check model info in expandable sections
        
        **Supported formats:** JPG, PNG, BMP, TIFF
        """)
        
        st.header("🔬 Knowledge Distillation")
        st.markdown("""
        The student model learns from the teacher using:
        - **Token-level KD** (70%): Match output distributions
        - **Feature KD** (20%): Match encoder representations
        - **Hidden state KD** (10%): Match internal states
        
        This creates a smaller, faster model while preserving most of the teacher's capabilities.
        """)
        
        st.header("⚡ Tips")
        st.markdown("""
        - Clear, well-lit images work best
        - Models trained on Flickr8k dataset
        - Common objects and scenes get better captions
        - Compare speed vs accuracy trade-offs
        - Student model is ideal for mobile deployment
        """)

if __name__ == "__main__":
    main()
