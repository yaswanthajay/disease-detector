# app.py
import streamlit as st
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image
import soundfile as sf
import librosa
import librosa.display
from scipy import signal
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model

# Optional LLM runtime
try:
    from llama_cpp import Llama
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Multi-Modal Health Analyzer")

# ==========================
# MultiModalHealthAnalyzer
# ==========================
class MultiModalHealthAnalyzer:
    def __init__(self):
        # Databases
        self.heart_conditions = self.create_heart_condition_database()
        self.eye_conditions = self.create_comprehensive_eye_database()

        # Models (lightweight architectures; in production load trained weights)
        self.heart_model = self.build_heart_sound_model()
        self.eye_model = self.build_advanced_eye_model()

        # Parameters
        self.confidence_threshold = 0.85
        self.min_audio_length = 3  # seconds
        self.sample_rate = 22050

    def create_heart_condition_database(self):
        return {
            0: {'name': 'Normal Heart Sound','description': 'Regular lub-dub pattern','urgency': 'None','icon': '❤️','color': '#4CAF50'},
            1: {'name': 'Heart Murmur','description': 'Abnormal whooshing sound','urgency': 'Consult cardiologist','icon': '💓','color': '#FF9800'},
            2: {'name': 'Arrhythmia','description': 'Irregular heart rhythm','urgency': 'See doctor soon','icon': '💔','color': '#F44336'},
            3: {'name': 'Gallop Rhythm','description': 'Extra heart sounds','urgency': 'Medical evaluation needed','icon': '🏇','color': '#9C27B0'}
        }

    def build_heart_sound_model(self):
        inputs = Input(shape=(128, 128, 3))
        x = layers.Conv2D(16, (3,3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(32, (3,3), activation='relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(len(self.heart_conditions), activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_heart_sound(self, audio_data):
        try:
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            # Resample if sample rate differs (we assume soundfile returns original SR)
            # Here we expect audio already at self.sample_rate or close enough
            duration = len(audio_data) / float(self.sample_rate)
            if duration < self.min_audio_length:
                raise ValueError(f"Audio too short. Needs at least {self.min_audio_length} seconds")

            f, t, Sxx = signal.spectrogram(audio_data, fs=self.sample_rate)
            # Normalize
            Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx) + 1e-9)
            Sxx = np.stack([Sxx]*3, axis=-1)
            Sxx = cv2.resize(Sxx, (128, 128))
            return np.expand_dims(Sxx, axis=0), None
        except Exception as e:
            return None, f"Audio processing error: {str(e)}"

    def create_heart_sound_visualization(self, audio_data):
        fig, axes = plt.subplots(2,1, figsize=(8,6))
        t = np.linspace(0, len(audio_data)/self.sample_rate, len(audio_data))
        axes[0].plot(t, audio_data)
        axes[0].set_title('Heart Sound Waveform')
        Sxx = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(Sxx, sr=self.sample_rate, x_axis='time', y_axis='log', ax=axes[1])
        axes[1].set_title('Spectrogram (dB)')
        plt.tight_layout()
        return fig

    def analyze_heart_sound(self, file_bytes):
        try:
            audio_data, sr = sf.read(io.BytesIO(file_bytes))
            # If sample rate differs, resample to self.sample_rate
            if sr != self.sample_rate:
                audio_data = librosa.resample(audio_data.astype(float), orig_sr=sr, target_sr=self.sample_rate)
            processed, err = self.preprocess_heart_sound(audio_data)
            if processed is None:
                return {'status':'error', 'error':err, 'recommendation':'Please try another recording'}
            pred = self.heart_model.predict(processed)[0]
            idx = int(np.argmax(pred))
            confidence = float(pred[idx])
            condition = self.heart_conditions.get(idx, {'name':'Unknown','description':'','urgency':'Consult specialist','icon':'❓','color':'#9E9E9E'})
            fig = self.create_heart_sound_visualization(audio_data)
            return {'status':'complete','diagnosis':condition['name'],'confidence':f"{confidence*100:.1f}%",'condition':condition,'visualization_fig':fig}
        except Exception as e:
            return {'status':'error','error':str(e),'recommendation':'Please try another recording'}

    def create_comprehensive_eye_database(self):
        return {
            'normal': {'name':'Healthy Eye','message':'No abnormalities detected','icon':'✅','color':'#4CAF50','advice':'Regular checkups recommended'},
            'conditions': {
                0: {'name':'Bacterial Conjunctivitis','type':'Infection','color':'#FF5722'},
                1: {'name':'Viral Conjunctivitis','type':'Viral Infection','color':'#F44336'},
                2: {'name':'Allergic Conjunctivitis','type':'Allergic Reaction','color':'#9C27B0'},
                3: {'name':'Dry Eye Syndrome','type':'Tear Film Disorder','color':'#FF9800'},
                4: {'name':'Corneal Abrasion','type':'Eye Injury','color':'#D32F2F'}
            }
        }

    def build_advanced_eye_model(self):
        inputs = Input(shape=(256,256,3))
        x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        att = layers.Conv2D(1,(1,1),activation='sigmoid')(x)
        x = layers.multiply([x, att])
        x = layers.Flatten()(x)
        x = layers.Dense(256,activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(len(self.create_comprehensive_eye_database()['conditions']), activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def analyze_eye_image(self, file_bytes, filename):
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            img_resized = img.resize((256,256))
            arr = np.array(img_resized) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = self.eye_model.predict(arr)[0]
            idx = int(np.argmax(pred))
            confidence = float(pred[idx])
            condition = self.create_comprehensive_eye_database()['conditions'].get(idx, {})
            return {'status':'complete','diagnosis':condition.get('name','Unknown'),'confidence':f"{confidence*100:.1f}%",'condition':condition}
        except Exception as e:
            return {'status':'error','error':str(e),'recommendation':'Please try another image'}

# ==========================
# Streamlit UI and logic
# ==========================
@st.cache_resource
def get_analyzer():
    return MultiModalHealthAnalyzer()

analyzer = get_analyzer()

st.markdown("<h1 style='text-align:center;color:#1a73e8;'>🩺 Multi-Modal Health Diagnosis</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    mode = st.radio("Select Mode", options=['Eye Analysis', 'Heart Sound Analysis'])
    st.markdown("---")
    if mode == 'Eye Analysis':
        st.info("Upload a clear eye image (close-up of eye, good lighting).")
        uploaded = st.file_uploader("Upload Eye Image", type=['png','jpg','jpeg','bmp'])
        # Also allow camera input (browser)
        camera_img = st.camera_input("Or capture from camera")
        file_bytes = None
        filename = None
        if camera_img is not None:
            file_bytes = camera_img.getvalue()
            filename = camera_img.name or "camera_capture.png"
        elif uploaded is not None:
            file_bytes = uploaded.read()
            filename = uploaded.name
    else:
        st.info("Upload or record heart sound (.wav, .mp3, .flac). Recordings: quiet room, 3+ seconds.")
        uploaded = st.file_uploader("Upload Heart Sound", type=['wav','mp3','flac','ogg'])
        st.markdown("**Tip:** You can upload a pre-recorded stethoscope or chest audio file.")
        file_bytes = None
        filename = None
        if uploaded is not None:
            file_bytes = uploaded.read()
            filename = uploaded.name

    run_btn = st.button("Analyze")

with col2:
    output_area = st.empty()

if run_btn:
    if not file_bytes:
        st.warning("Please upload or capture a file first.")
    else:
        with st.spinner("Processing..."):
            if mode == 'Eye Analysis':
                result = analyzer.analyze_eye_image(file_bytes, filename or "eye.jpg")
                if result['status'] == 'error':
                    st.error(result.get('error'))
                else:
                    st.success(f"Diagnosis: {result['diagnosis']}  —  Confidence: {result['confidence']}")
                    condition = result.get('condition', {})
                    st.markdown(f"**Type:** {condition.get('type','N/A')}")
                    st.markdown(f"**Color indicator:** <span style='background:{condition.get('color','#ccc')}; padding:4px 8px; border-radius:4px;'>{condition.get('name','')}</span>", unsafe_allow_html=True)
                    # show image preview
                    st.image(file_bytes, caption="Input Image", use_column_width=True)
            else:
                result = analyzer.analyze_heart_sound(file_bytes)
                if result['status'] == 'error':
                    st.error(result.get('error'))
                    st.write(result.get('recommendation',''))
                else:
                    st.success(f"Diagnosis: {result['diagnosis']}  —  Confidence: {result['confidence']}")
                    cond = result.get('condition', {})
                    st.markdown(f"**Description:** {cond.get('description','-')}")
                    # visualization
                    fig = result.get('visualization_fig')
                    if fig:
                        st.pyplot(fig)
                    # play audio
                    try:
                        st.audio(file_bytes)
                    except Exception:
                        st.info("Audio playback not supported for this format.")

# Small footer with model notes and optional LLM form
st.markdown("---")
st.markdown("**Notes:** This is a demo pipeline. Replace model architectures with trained weights for production. If using a local LLM via `llama-cpp-python`, ensure the model file path and installation are correct.")
if LLM_AVAILABLE:
    st.success("LLM runtime detected (`llama-cpp-python`). You can wire in text-generation later.")
else:
    st.info("LLM runtime not installed — text generation disabled.")
