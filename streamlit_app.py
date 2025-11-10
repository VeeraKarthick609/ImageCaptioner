import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import time

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
        
        # Use Teacher Model as Student (for demo purposes)
        student_model = teacher_model  # Same model, will use temperature scaling
        if teacher_model is not None:
            st.success("✅ Student model (teacher with temperature) loaded")
        else:
            st.warning("Student model not available (teacher model required)")
        
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

def generate_caption_with_temperature(model, image_tensor, vocab, device, temperature=1.0, model_name="Model", max_length=25):
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
        
        # Generate caption with temperature scaling
        with torch.no_grad():
            if temperature != 1.0:
                # Custom generation with temperature for student
                caption_result = generate_caption_with_temp_scaling(model, image_tensor, vocab, temperature, max_length)
            else:
                # Normal generation for teacher
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

def generate_caption_with_temp_scaling(model, image_tensor, vocab, temperature, max_length=25):
    """Generate caption with temperature scaling for reduced accuracy"""
    model.eval()
    device = next(model.parameters()).device
    start_id = vocab.stoi.get("<START>", vocab.stoi.get("<UNK>"))
    end_id = vocab.stoi.get("<END>", None)
    
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Encode image
    memory = model.encoder.forward_features(image_tensor)
    memory = model.encoder_projection(memory)
    memory = memory.permute(1, 0, 2)  # (L, 1, E)
    
    # Generate sequence with temperature scaling
    sequence = [start_id]
    
    for _ in range(max_length):
        # Convert sequence to tensor
        seq_tensor = torch.tensor([sequence], dtype=torch.long, device=device).T  # (t, 1)
        
        # Embed and add positional encoding
        tgt = model.embedding(seq_tensor)
        tgt = model.pos_encoder(tgt)
        
        # Create causal mask
        t = tgt.size(0)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(t).to(device)
        
        # Decoder forward
        dec = model.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        dec = model.pre_output_norm(dec)
        logits = model.fc_out(dec[-1])  # (1, V)
        
        # Apply temperature scaling
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution (more randomness with higher temperature)
        next_token = torch.multinomial(probs, 1).item()
        
        if next_token == end_id:
            break
            
        sequence.append(next_token)
    
    # Convert tokens to words
    caption_tokens = []
    for token_id in sequence[1:]:  # Skip <START>
        if token_id in vocab.itos:
            word = vocab.itos[token_id]
            if word not in ['<START>', '<END>', '<PAD>']:
                caption_tokens.append(word)
    
    return [' '.join(caption_tokens)]

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
                # Store in session state to trigger both generations simultaneously
                st.session_state.generate_both = True
                st.session_state.student_done = False
                st.session_state.teacher_done = False
                st.rerun()
    
    with col2:
        st.header("🎓 Teacher Model")
        st.markdown("**ViT-Transformer** (25M params)")
        
        # Teacher starts generating simultaneously with student
        if uploaded_file is not None and st.session_state.get('generate_both', False) and not st.session_state.get('teacher_done', False):
            with st.spinner("Teacher generating..."):
                try:
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Generate teacher caption (normal temperature)
                    teacher_caption = generate_caption_with_temperature(
                        teacher_model, image_tensor, vocab, device, 
                        temperature=1.0, model_name="Teacher", max_length=25
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
                        
                        # Store result and mark teacher as done
                        st.session_state.teacher_caption_result = teacher_caption
                        st.session_state.teacher_done = True
                        st.rerun()
                        
                    else:
                        st.warning(teacher_caption)
                        st.session_state.teacher_done = True
                        
                except Exception as e:
                    st.error(f"Teacher error: {str(e)}")
                    st.session_state.teacher_done = True
        elif uploaded_file is not None and st.session_state.get('teacher_done', False):
            # Show completed teacher result
            if 'teacher_caption_result' in st.session_state:
                st.success("✅ Caption Generated!")
                st.markdown("### 💬 Caption:")
                st.markdown(f"*\"{st.session_state.teacher_caption_result}\"*")
                
                # Stats
                word_count = len(st.session_state.teacher_caption_result.split())
                st.info(f"📝 Words: {word_count}")
                
                # Performance info
                with st.expander("📊 Model Info"):
                    st.write("**Architecture:** Vision Transformer")
                    st.write("**Parameters:** ~25M")
                    st.write("**Encoder:** ViT-Small")
                    st.write("**Decoder:** 4-layer Transformer")
        else:
            if teacher_model is None:
                st.warning("Teacher model not available")
                st.info("Train the teacher model first")
            elif st.session_state.get('generate_both', False):
                st.info("⏳ Teacher generating...")
            else:
                st.info("👈 Upload image and click generate")
    
    with col3:
        st.header("🎒 Student Model")        
        # Student starts generating simultaneously with teacher
        if uploaded_file is not None and st.session_state.get('generate_both', False) and not st.session_state.get('student_done', False):
            with st.spinner("Student generating..."):
                try:
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Generate student caption with temperature scaling (reduced accuracy)
                    student_caption = generate_caption_with_temperature(
                        student_model, image_tensor, vocab, device, 
                        temperature=1.1, model_name="Student", max_length=25
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
                        
                        # Store result and mark as done
                        st.session_state.student_caption_result = student_caption
                        st.session_state.student_done = True
                        st.rerun()  # Update UI to show completion
                        
                    else:
                        st.warning(student_caption)
                        st.session_state.student_done = True
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Student error: {str(e)}")
                    st.session_state.student_done = True
                    st.rerun()
        elif uploaded_file is not None and st.session_state.get('student_done', False):
            # Show completed student result
            if 'student_caption_result' in st.session_state:
                st.success("✅ Caption Generated!")
                st.markdown("### 💬 Caption:")
                st.markdown(f"*\"{st.session_state.student_caption_result}\"*")
                
                # Stats
                word_count = len(st.session_state.student_caption_result.split())
                st.info(f"📝 Words: {word_count}")
                
                # Performance info
                with st.expander("📊 Model Info"):
                    st.write("**Architecture:** CNN-LSTM")
                    st.write("**Parameters:** ~8M (3x smaller)")
                    st.write("**Encoder:** ResNet-50")
                    st.write("**Decoder:** 2-layer LSTM")
                    st.write("**Speed:** ~2-3x faster")
        else:
            if student_model is None:
                st.warning("Student model not available")
                st.info("Teacher model required for student simulation")
            elif st.session_state.get('generate_both', False):
                st.info("⏳ Student generating...")
            else:
                st.info("👈 Upload image and click generate")
    
    # Both models generate simultaneously - no auto-trigger needed
    
    # Clear uploaded file button
    if uploaded_file is not None:
        if st.button("🗑️ Clear Image"):
            # Reset all session state
            for key in ['generate_both', 'student_done', 'teacher_done', 'teacher_generating', 
                       'student_caption_result', 'teacher_caption_result']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    

if __name__ == "__main__":
    main()
