# 🌿 Medicinal Leaf Identification

A deep learning-based web application that identifies **80 types of medicinal leaves** from images using a VGG16 transfer learning model and a Streamlit interface.

## Features
- Identifies 80 medicinal plant species from leaf images
- Built with TensorFlow/Keras and Streamlit
- Works on both **Local System** and **Google Colab**

## Setup

### Option 1: Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Medicinal_Leaf_Identification.git
   cd Medicinal_Leaf_Identification
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the trained model:**
   - Download `leaf_model.keras` from [Google Drive](YOUR_DRIVE_LINK_HERE)
   - Place it in the project root folder
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

### Option 2: Run on Google Colab
1. Open a new Colab notebook.
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Install Streamlit:
   ```bash
   !pip install -q streamlit
   ```
4. Copy `app.py` to Colab working directory:
   ```bash
   !cp /content/drive/MyDrive/Medicinal_Leaf_Identification/app.py .
   ```
5. Make sure `leaf_model.keras` is in your Google Drive root (`/content/drive/MyDrive/leaf_model.keras`).
6. Run the app:
   ```bash
   !streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false & npx --yes localtunnel --port 8501
   ```

## Model Training
To train the model from scratch, use `train.py` with the [Indian Medicinal Leaves Image Dataset](https://www.kaggle.com/).

```bash
python train.py
```

## Tech Stack
- **Model:** VGG16 (Transfer Learning)
- **Framework:** TensorFlow / Keras
- **UI:** Streamlit
- **Language:** Python

## Project Structure
```
Medicinal_Leaf_Identification/
├── app.py                  # Streamlit web application
├── train.py                # Model training script
├── requirements.txt        # Python dependencies
├── leaf_model.keras        # Trained model (not in repo, download separately)
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## License
This project is for educational purposes.
