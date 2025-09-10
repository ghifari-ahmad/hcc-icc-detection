import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# --- Pengaturan Halaman dan Judul ---
st.set_page_config(
    page_title="Deteksi Penyakit Hati (HCC & ICC)",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Aplikasi Segmentasi YOLO untuk Deteksi HCC & ICC")
st.write(
    "Unggah gambar medis Anda untuk melakukan segmentasi objek dan mendeteksi "
    "potensi **Hepatocellular Carcinoma (HCC)** dan **Intrahepatic Cholangiocarcinoma (ICC)**."
)

# --- Fungsi Memuat Model ---
@st.cache_resource
def load_yolo_model(model_path):
    """Memuat model YOLO dari path yang diberikan."""
    model = YOLO(model_path)
    return model

# --- Sidebar untuk Input Pengguna ---
with st.sidebar:
    st.header("⚙️ Pengaturan")

    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

    confidence_level = st.slider(
        "Atur Confidence Level",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05
    )

    st.info(
        f"Hanya deteksi dengan tingkat kepercayaan **≥ {confidence_level:.2f}** yang akan ditampilkan."
    )

# --- Area Tampilan Utama ---
MODEL_PATH = "best.pt" 
model = load_yolo_model(MODEL_PATH)

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Gambar Asli")
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

    with col2:
        st.subheader("🎨 Hasil Segmentasi")
        with st.spinner("Menganalisis gambar dengan YOLO..."):
            results = model.predict(source=image, conf=confidence_level)
            result_plot = results[0].plot()
            result_plot_rgb = cv2.cvtColor(result_plot, cv2.COLOR_BGR2RGB)

        st.image(
            result_plot_rgb,
            caption=f"Hasil Segmentasi dengan Confidence ≥ {confidence_level:.2f}",
            use_column_width=True
        )

elif model is None:
    st.warning("Model tidak berhasil dimuat. Silakan periksa path dan file model Anda.")
else:
    st.info("Silakan unggah gambar melalui panel di sebelah kiri untuk memulai analisis.")
