# app_optimized.py

import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import tempfile 

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Emosi Suara (Optimized)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Path dan Label ---
TFLITE_MODEL_PATH = os.path.join('models', 'model.tflite') 
EMOTION_LABELS = sorted(['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'])

# --- Fungsi-fungsi Aplikasi yang Sudah Dioptimalkan ---

@st.cache_resource
def load_tflite_model():
    """Memuat model TFLite dan mengalokasikan tensor."""
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error saat memuat model TFLite di '{TFLITE_MODEL_PATH}': {e}")
        st.info("Pastikan Anda sudah menjalankan skrip 'convert_to_tflite.py' terlebih dahulu.")
        return None

@st.cache_data
def extract_features_from_path(file_path, sr=22050, n_mfcc=40, max_pad_len=862):
    """Mengekstrak fitur MFCC dari path file audio."""
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=3.0)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        st.error(f"Error saat memproses file audio: {e}")
        return None

def get_feedback(label):
    """Memberikan feedback berdasarkan label emosi dominan."""
    feedback_dict = {
        'angry': "🔥 Terdeteksi marah. Suara Anda terdengar tegas. Untuk beberapa konteks, ini bagus untuk penekanan. Namun, untuk audiens umum, coba bicara dengan intonasi lebih kalem dan hindari tekanan berlebih.",
        'happy': "😊 Bagus! Suaramu terdengar menyenangkan dan bersemangat. Energi positif seperti ini sangat menular ke audiens. Pertahankan!",
        'sad': "💧 Terdeteksi sedih. Jika ini disengaja untuk story-telling, ini efektif. Jika tidak, coba tingkatkan semangat dan naikkan intonasi untuk menjaga perhatian audiens.",
        'neutral': "😐 Netral. Suara Anda terdengar jelas namun kurang menunjukkan emosi. Coba tambahkan lebih banyak variasi intonasi (naik-turun) agar tidak terdengar datar dan lebih menarik.",
        'fearful': "😨 Terdeteksi takut atau gugup. Ini wajar. Tenangkan diri sebelum bicara, ambil napas dalam, dan latih artikulasi Anda. Berbicara sedikit lebih lambat bisa membantu.",
        'calm': "🧘 Sangat baik! Suara Anda terdengar tenang, terkontrol, dan nyaman untuk didengar. Ini menciptakan suasana yang meyakinkan dan dapat dipercaya.",
        'surprised': "😮 Terdengar terkejut. Emosi ini sangat bagus untuk digunakan sebagai penekanan pada poin-poin penting atau saat menyampaikan sesuatu yang tidak terduga untuk membangun klimaks.",
        'disgust': "🤢 Terdeteksi emosi negatif (jijik/tidak suka). Hati-hati dengan nada ini karena bisa membuat audiens merasa tidak nyaman atau tersinggung. Pastikan nada Anda sesuai dengan pesan yang ingin disampaikan.",
    }
    return feedback_dict.get(label, "Latih lagi artikulasi dan emosi agar lebih meyakinkan.")

def run_tflite_inference(interpreter, input_data):
    """Menjalankan prediksi menggunakan interpreter TFLite."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# --- Tampilan Aplikasi (UI) ---

interpreter = load_tflite_model()

st.title("⚡ AI Public Speaking Coach")
st.markdown("Aplikasi ini sudah dioptimalkan untuk analisis yang lebih cepat menggunakan TensorFlow Lite.")

st.sidebar.header("Upload Audio Anda")
audio_file = st.sidebar.file_uploader("Pilih file audio (.wav)", type=["wav"])

# --- INI BAGIAN YANG DITAMBAHKAN ---
st.sidebar.markdown("---")
st.sidebar.markdown("Crafted with by **Raihan & Reynanda**")
# ------------------------------------

if not interpreter:
    st.warning("Aplikasi belum siap. Model TFLite tidak dapat dimuat.")
else:
    if audio_file is not None:
        st.header("Hasil Analisis", divider='rainbow')
        st.audio(audio_file, format="audio/wav")

        with st.spinner("Menganalisis dengan kecepatan kilat... ⚡"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
                fp.write(audio_file.getvalue())
                temp_path = fp.name
            
            features = extract_features_from_path(temp_path)
            os.remove(temp_path)

            if features is not None:
                input_data = features[np.newaxis, ..., np.newaxis].astype(np.float32)
                prediction = run_tflite_inference(interpreter, input_data)

                # --- Tampilan Hasil ---
                col1, col2 = st.columns(2, gap="large")

                with col1:
                    st.subheader("🧠 Emosi Dominan")
                    idx_max = np.argmax(prediction)
                    emosi_dominan = EMOTION_LABELS[idx_max]
                    confidence = prediction[idx_max]
                    st.metric(
                        label="Emosi Terkuat Terdeteksi",
                        value=emosi_dominan.capitalize(),
                        delta=f"{confidence*100:.2f}% Keyakinan"
                    )
                    
                    st.markdown("### 💡 Saran untuk Anda")
                    feedback = get_feedback(emosi_dominan)
                    st.success(feedback, icon="✅")
                
                with col2:
                    st.subheader("📈 Distribusi Emosi")
                    df_emosi = pd.DataFrame({
                        'Probabilitas': prediction,
                        'Emosi': [label.capitalize() for label in EMOTION_LABELS]
                    })
                    df_emosi = df_emosi.set_index('Emosi')
                    st.bar_chart(df_emosi)
    else:
        st.info("Silakan upload file audio untuk memulai analisis.")
