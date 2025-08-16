import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        height: 20px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Ensure RGB format
    image = image.convert("RGB")
    
    # Resize to 150x150
    img_resized = image.resize((150, 150))
    
    # Convert to numpy and normalize
    img_array = np.array(img_resized)
    img_normalized = img_array / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_array

def predict_image(model, image):
    """Make prediction on the image"""
    try:
        # Preprocess image
        img_batch, img_rgb = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(img_batch, verbose=0)
        
        # Get class probabilities
        cat_prob = prediction[0][0]
        dog_prob = prediction[0][1]
        
        # Determine class
        if cat_prob > dog_prob:
            predicted_class = "Cat"
            confidence = cat_prob
        else:
            predicted_class = "Dog"
            confidence = dog_prob
            
        return predicted_class, confidence, cat_prob, dog_prob, img_rgb
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🐱 Cat vs Dog Classifier 🐶</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading the trained model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please make sure 'model.h5' file is in the current directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.title("📋 Instructions")
    st.sidebar.markdown("""
    1. **Upload an image** of a cat or dog
    2. **Wait for prediction** - the model will analyze the image
    3. **View results** - see the predicted class and confidence
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Info:**")
    st.sidebar.markdown("- Input size: 150x150 pixels")
    st.sidebar.markdown("- Classes: Cat, Dog")
    st.sidebar.markdown("- Architecture: CNN with 4 Conv layers")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a cat or dog"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Prediction button
            if st.button("🔍 Predict", type="primary"):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, cat_prob, dog_prob, processed_img = predict_image(model, image)
                
                if predicted_class is not None:
                    # Display results
                    with col2:
                        st.subheader("🎯 Prediction Results")
                        
                        # Prediction box
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Predicted: <strong>{predicted_class}</strong></h3>
                            <p>Confidence: <strong>{confidence:.2%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bars
                        st.markdown("**Detailed Probabilities:**")
                        
                        # Cat probability
                        st.markdown("🐱 Cat:")
                        cat_percent = cat_prob * 100
                        st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {cat_percent}%"></div>
                        </div>
                        <p>{cat_prob:.2%}</p>
                        """, unsafe_allow_html=True)
                        
                        # Dog probability
                        st.markdown("🐶 Dog:")
                        dog_percent = dog_prob * 100
                        st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {dog_percent}%"></div>
                        </div>
                        <p>{dog_prob:.2%}</p>
                        """, unsafe_allow_html=True)
                        
                        # Display processed image
                        st.subheader("🖼️ Processed Image")
                        st.image(processed_img, caption="Image after preprocessing (150x150)", use_column_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
        <p>Model trained on Cats vs Dogs dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 