import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Medicinal Leaf Identification", layout="centered")
st.title("🌿 Medicinal Leaf Identification")


# 1. AUTO-DETECT ENVIRONMENT (Local or Google Colab)
def is_colab():
    """Check if we are running inside Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


@st.cache_resource
def load_model():
    """Load model from the correct path based on environment."""
    if is_colab():
        # Google Colab: model is stored in Google Drive
        model_path = '/content/drive/MyDrive/leaf_model.keras'
    else:
        # Local system: model is in the same folder as app.py
        model_path = os.path.join(os.path.dirname(__file__), 'leaf_model.keras')

    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found at: `{model_path}`\n\n"
                 "Please make sure `leaf_model.keras` is in the correct location.")
        st.stop()

    return tf.keras.models.load_model(model_path)


model = load_model()

# 2. CLASS NAMES (80 medicinal plant species)
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

# 3. APP UI
uploaded_file = st.file_uploader("Upload a medicinal leaf image for identification...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)

    # Preprocessing to match training model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button('🚀 Identify Leaf'):
        with st.spinner('Analyzing the qualities of this leaf...'):
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions[0])
            confidence = 100 * np.max(predictions[0])

            st.success(f"Result: This leaf is identified as **{CLASS_NAMES[class_idx]}**")
            st.info(f"Identity Confidence: {confidence:.2f}%")

st.markdown("---")
env_label = "☁️ Google Colab" if is_colab() else "💻 Local System"
st.caption(f"Running on: {env_label} | Medicinal Leaf Identification Project")
