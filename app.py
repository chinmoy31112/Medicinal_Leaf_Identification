import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Medicinal Leaf Identification", layout="centered")

st.title("🌿 Medicinal Leaf Identification")
st.write("Upload an image of a medicinal leaf to identify its type.")

# Define class names (80 classes found in the dataset)
CLASS_NAMES = [
    'Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'ashoka', 'Astma_weed', 'Badipala', 'Balloon_Vine', 
    'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'camphor', 'Caricature', 'Castor', 
    'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 
    'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 
    'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 
    'Jackfruit', 'Jasmine', 'kamakasturi', 'Kambajala', 'Kasambruga', 'kepala', 'Kohlrabi', 
    'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 
    'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 
    'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 
    'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 
    'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric'
]

# Load model
@st.cache_resource
def load_model():
    model_path = 'leaf_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

if model is None:
    st.warning("⚠️ Model not found! Please run the training notebook first to save 'leaf_model.keras'.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button('Identify Leaf'):
            with st.spinner('Analyzing...'):
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                class_idx = np.argmax(predictions[0])
                confidence = 100 * np.max(predictions[0])
                
                st.success(f"Prediction: **{CLASS_NAMES[class_idx]}**")
                st.info(f"Confidence: {confidence:.2f}%")

st.markdown("---")
st.write("Developed for Medicinal Leaf Identification Project")
