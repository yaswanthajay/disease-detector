import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Image as IPImage
import io
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import base64
from matplotlib import patches
import soundfile as sf
import librosa
import librosa.display
from scipy import signal
from IPython.display import Javascript

class MultiModalHealthAnalyzer:
    def __init__(self):
        # Initialize databases first
        self.heart_conditions = self.create_heart_condition_database()
        self.eye_conditions = self.create_comprehensive_eye_database()

        # Then build models that depend on these databases
        self.heart_model = self.build_heart_sound_model()
        self.eye_model = self.build_advanced_eye_model()

        # Other parameters
        self.confidence_threshold = 0.85
        self.min_eye_size = 2500
        self.min_focus_measure = 60
        self.min_audio_length = 3  # seconds
        self.sample_rate = 22050

    # ====================== Heart Sound Analysis ======================
    def create_heart_condition_database(self):
        """Database of heart conditions detectable from sounds"""
        return {
            0: {
                'name': 'Normal Heart Sound',
                'description': 'Regular lub-dub pattern with normal intervals',
                'urgency': 'None',
                'icon': '❤️',
                'color': '#4CAF50'
            },
            1: {
                'name': 'Heart Murmur',
                'description': 'Abnormal whooshing or swishing sound between beats',
                'urgency': 'Consult cardiologist',
                'icon': '💓',
                'color': '#FF9800'
            },
            2: {
                'name': 'Arrhythmia',
                'description': 'Irregular heart rhythm, possible skipped beats',
                'urgency': 'See doctor soon',
                'icon': '💔',
                'color': '#F44336'
            },
            3: {
                'name': 'Gallop Rhythm',
                'description': 'Extra heart sounds creating a galloping rhythm',
                'urgency': 'Medical evaluation needed',
                'icon': '🏇',
                'color': '#9C27B0'
            }
        }

    def build_heart_sound_model(self):
        """CNN model for heart sound classification"""
        inputs = Input(shape=(128, 128, 3))  # For spectrogram images

        x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(64, (3,3), activation='relu')(x)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(len(self.heart_conditions), activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model

    def preprocess_heart_sound(self, audio_data):
        """Convert audio to spectrogram and preprocess"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample if needed
            if len(audio_data) / self.sample_rate < self.min_audio_length:
                raise ValueError(f"Audio too short. Needs at least {self.min_audio_length} seconds")

            # Create spectrogram
            f, t, Sxx = signal.spectrogram(audio_data, fs=self.sample_rate)

            # Convert to image-like format
            Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
            Sxx = np.stack([Sxx]*3, axis=-1)  # Convert to 3-channel
            Sxx = cv2.resize(Sxx, (128, 128))

            return np.expand_dims(Sxx, axis=0), None  # Return processed spectrogram and visualization

        except Exception as e:
            return None, f"Audio processing error: {str(e)}"

    def analyze_heart_sound(self, file_content):
        """Analyze uploaded heart sound recording"""
        try:
            # Read audio file
            audio_data, _ = sf.read(io.BytesIO(file_content))

            # Preprocess and create spectrogram
            processed_audio, error = self.preprocess_heart_sound(audio_data)
            if processed_audio is None:
                return {
                    'status': 'error',
                    'error': error,
                    'recommendation': 'Please try another recording'
                }

            # Make prediction
            pred = self.heart_model.predict(processed_audio)[0]
            pred_idx = np.argmax(pred)
            confidence = pred[pred_idx]

            # Get condition details
            condition = self.heart_conditions.get(pred_idx, {
                'name': 'Unknown Condition',
                'description': 'Unclassified heart sound',
                'urgency': 'Consult specialist',
                'icon': '❓',
                'color': '#9E9E9E'
            })

            # Create visualization
            fig = self.create_heart_sound_visualization(audio_data)

            return {
                'status': 'complete',
                'diagnosis': condition['name'],
                'confidence': f"{confidence*100:.1f}%",
                'condition': condition,
                'visualization': fig
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': f"Analysis failed: {str(e)}",
                'recommendation': 'Please try another recording'
            }

    def create_heart_sound_visualization(self, audio_data):
        """Create visualization of heart sound waveform and spectrogram"""
        plt.figure(figsize=(12, 8))

        # Waveform
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, len(audio_data)/self.sample_rate, len(audio_data)), audio_data)
        plt.title('Heart Sound Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        # Spectrogram
        plt.subplot(2, 1, 2)
        Sxx = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(Sxx, sr=self.sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')

        plt.tight_layout()
        return plt.gcf()

    # ====================== Eye Analysis ======================
    def create_comprehensive_eye_database(self):
        """Expanded database with more conditions and details"""
        return {
            'normal': {
                'name': 'Healthy Eye',
                'message': 'No abnormalities detected',
                'icon': '✅',
                'color': '#4CAF50',
                'advice': 'Regular eye checkups recommended'
            },
            'conditions': {
                0: {'name': 'Bacterial Conjunctivitis', 'type': 'Infection', 'color': '#FF5722'},
                1: {'name': 'Viral Conjunctivitis', 'type': 'Viral Infection', 'color': '#F44336'},
                2: {'name': 'Allergic Conjunctivitis', 'type': 'Allergic Reaction', 'color': '#9C27B0'},
                3: {'name': 'Dry Eye Syndrome', 'type': 'Tear Film Disorder', 'color': '#FF9800'},
                4: {'name': 'Corneal Abrasion', 'type': 'Eye Injury', 'color': '#D32F2F'}
            }
        }

    def build_advanced_eye_model(self):
        """Enhanced CNN model with attention mechanism"""
        inputs = Input(shape=(256, 256, 3))

        # Feature extraction backbone
        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)

        # Attention mechanism
        attention = layers.Conv2D(1, (1,1), activation='sigmoid')(x)
        x = layers.multiply([x, attention])

        # Classification head
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(len(self.eye_conditions['conditions']), activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model

    def analyze_eye_image(self, file_content, filename):
        """Comprehensive eye analysis"""
        try:
            # Mock preprocessing - in a real app this would process the image
            processed_img = np.random.random((1, 256, 256, 3))  # Mock processing

            # Mock prediction
            pred = self.eye_model.predict(processed_img)[0]
            pred_idx = np.argmax(pred)
            confidence = pred[pred_idx]

            condition = self.eye_conditions['conditions'].get(pred_idx, {})

            return {
                'status': 'complete',
                'diagnosis': condition.get('name', 'Unknown'),
                'confidence': f"{confidence*100:.1f}%",
                'condition': condition
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'recommendation': 'Please try another image'
            }

class HealthDiagnosisDashboard:
    def __init__(self):
        self.analyzer = MultiModalHealthAnalyzer()
        self.uploaded_file = None
        self.current_mode = 'eye'  # 'eye' or 'heart'
        self.camera_active = False

        # UI Components
        self.mode_selector = widgets.RadioButtons(
            options=[('👁️ Eye Analysis', 'eye'), ('❤️ Heart Sound Analysis', 'heart')],
            description='Analysis Mode:',
            disabled=False
        )

        self.file_upload = widgets.FileUpload(
            description="Upload File",
            accept='',  # Will be set based on mode
            multiple=False,
            style={'description_width': 'initial'}
        )

        self.analyze_btn = widgets.Button(
            description="Analyze",
            button_style='success',
            icon='search',
            disabled=True
        )

        self.clear_btn = widgets.Button(
            description="Clear",
            button_style='warning',
            icon='trash'
        )

        self.camera_btn = widgets.Button(
            description="Capture Eye Image",
            button_style='info',
            icon='camera',
            disabled=False
        )

        self.record_btn = widgets.Button(
            description="Record Heart Sound",
            button_style='info',
            icon='microphone',
            disabled=False
        )

        self.video_output = widgets.Output(
            layout={'border': '1px solid black', 'width': '400px', 'height': '300px'}
        )

        self.capture_btn = widgets.Button(
            description="Take Snapshot",
            button_style='primary',
            disabled=True
        )

        self.content_display = widgets.Output(
            layout={'border': '1px solid #ddd', 'width': '800px', 'min_height': '400px'}
        )

        self.analysis_output = widgets.Output()

        # Event handlers
        self.mode_selector.observe(self.on_mode_change, names='value')
        self.file_upload.observe(self.on_upload, names='value')
        self.analyze_btn.on_click(self.run_analysis)
        self.clear_btn.on_click(self.clear_analysis)
        self.camera_btn.on_click(self.toggle_camera)
        self.record_btn.on_click(self.record_heart_sound)
        self.capture_btn.on_click(self.capture_image)

        # Initialize
        self.on_mode_change({'new': 'eye'})

        # Display UI
        display(widgets.VBox([
            widgets.HTML("<h1 style='color:#1a73e8; text-align:center;'>🩺 Multi-Modal Health Diagnosis</h1>"),
            self.mode_selector,
            widgets.HBox([
                widgets.VBox([
                    self.file_upload,
                    widgets.HBox([self.camera_btn, self.record_btn]),
                    widgets.HBox([self.analyze_btn, self.clear_btn]),
                    self.video_output,
                    self.capture_btn
                ]),
                self.content_display
            ]),
            self.analysis_output
        ]))

    def on_mode_change(self, change):
        """Handle mode selection change"""
        self.current_mode = change['new']
        if self.current_mode == 'eye':
            self.file_upload.accept = '.png,.jpg,.jpeg,.bmp'
            self.file_upload.description = "Upload Eye Image"
            self.record_btn.disabled = True
            self.camera_btn.disabled = False
        else:
            self.file_upload.accept = '.wav,.mp3,.ogg,.flac'
            self.file_upload.description = "Upload Heart Sound"
            self.record_btn.disabled = False
            self.camera_btn.disabled = True
            if self.camera_active:
                self.stop_camera()

        with self.content_display:
            clear_output()
            if self.current_mode == 'eye':
                display(HTML("""
                <p style='text-align:center; color:#666;'>
                    Upload a clear eye image or use camera for analysis<br>
                    <small>Close-up of the eye with good lighting works best</small>
                </p>
                """))
            else:
                display(HTML("""
                <p style='text-align:center; color:#666;'>
                    Upload or record heart sound for analysis<br>
                    <small>Record in quiet environment for best results</small>
                </p>
                """))

    def toggle_camera(self, btn):
        """Toggle camera on/off"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start web camera for eye image capture"""
        self.camera_active = True
        self.camera_btn.description = "Stop Camera"
        self.capture_btn.disabled = False

        display(Javascript('''
            const video = document.createElement('video');
            video.id = 'video';
            video.width = 400;
            video.height = 300;
            video.autoplay = true;

            const promise = navigator.mediaDevices.getUserMedia({ video: true });
            promise.then(function(stream) {
                video.srcObject = stream;
                const videoOutput = document.querySelector('.widget-output');
                videoOutput.appendChild(video);
            });

            window.cameraVideo = video;
        '''))

    def stop_camera(self):
        """Stop web camera"""
        self.camera_active = False
        self.camera_btn.description = "Capture Eye Image"
        self.capture_btn.disabled = True

        display(Javascript('''
            if (window.cameraVideo && window.cameraVideo.srcObject) {
                window.cameraVideo.srcObject.getTracks().forEach(track => track.stop());
                window.cameraVideo.remove();
            }
        '''))
        with self.video_output:
            clear_output()

    def capture_image(self, btn):
        """Capture image from camera"""
        display(Javascript('''
            const canvas = document.createElement('canvas');
            canvas.width = 400;
            canvas.height = 300;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(window.cameraVideo, 0, 0, canvas.width, canvas.height);

            const dataUrl = canvas.toDataURL('image/png');
            const byteString = atob(dataUrl.split(',')[1]);
            const mimeString = dataUrl.split(',')[0].split(':')[1].split(';')[0];
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);

            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }

            const blob = new Blob([ab], {type: mimeString});
            const file = new File([blob], "captured_image.png", {type: mimeString});

            // Store the file in a global variable to be accessed by Python
            window.capturedImageFile = file;

            // Display the captured image
            const img = document.createElement('img');
            img.src = dataUrl;
            img.width = 400;
            img.height = 300;

            const outputDiv = document.querySelector('.widget-output');
            outputDiv.innerHTML = '';
            outputDiv.appendChild(img);
        '''))

        # In a real implementation, we would need a way to get the image data back to Python
        # This is a limitation of Jupyter widgets - we can't directly access the captured image
        # For demo purposes, we'll just show a message
        with self.content_display:
            clear_output()
            display(HTML("""
            <div style='text-align:center; margin:20px;'>
                <p style='color:#1a73e8;'>Image captured from camera!</p>
                <p>In a real implementation, this image would be analyzed</p>
            </div>
            """))

        self.uploaded_file = {'content': b'mock_image_data', 'name': 'captured_image.png'}
        self.analyze_btn.disabled = False

    def on_upload(self, change):
        """Handle file upload with preview"""
        if change['new']:
            self.uploaded_file = next(iter(change['new'].values()))
            self.analyze_btn.disabled = False

            with self.content_display:
                clear_output()
                try:
                    if self.current_mode == 'eye':
                        img = Image.open(io.BytesIO(self.uploaded_file['content']))

                        fig, ax = plt.subplots(figsize=(8,6))
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title("Uploaded Eye Image", color='green')
                        plt.tight_layout()
                        plt.show()
                    else:
                        audio_data, _ = sf.read(io.BytesIO(self.uploaded_file['content']))

                        fig = plt.figure(figsize=(10,4))
                        plt.plot(audio_data)
                        plt.title("Uploaded Heart Sound Waveform")
                        plt.xlabel("Samples")
                        plt.ylabel("Amplitude")
                        plt.tight_layout()
                        plt.show()

                except Exception as e:
                    display(HTML(f"<p style='color:red;'>Error displaying content: {str(e)}</p>"))

    def record_heart_sound(self, btn):
        """Handle heart sound recording"""
        with self.content_display:
            clear_output()
            display(HTML("""
            <div style='text-align:center; margin:20px;'>
                <p style='color:#1a73e8;'>Recording functionality would be implemented here</p>
                <p>In a real application, this would use the browser's MediaRecorder API</p>
                <p>For this demo, please upload a pre-recorded heart sound file</p>
            </div>
            """))

    def run_analysis(self, btn):
        """Run analysis based on current mode"""
        if not self.uploaded_file:
            with self.analysis_output:
                display(HTML("<p style='color:red;'>Please upload or capture a file first</p>"))
            return

        with self.analysis_output:
            clear_output()
            display(HTML(f"<h3 style='color:#1a73e8;'>Analyzing {'Eye Image' if self.current_mode == 'eye' else 'Heart Sound'}...</h3>"))

            # Add loading animation
            display(HTML("""
            <div style="text-align:center; margin:20px 0;">
                <div style="font-size:24px; animation: spin 1s linear infinite;">🔍</div>
                <p>Processing and analyzing data...</p>
                <style>
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            </div>
            """))

            # Run appropriate analysis
            if self.current_mode == 'eye':
                result = self.analyzer.analyze_eye_image(
                    self.uploaded_file['content'],
                    self.uploaded_file.get('name', 'eye_image.jpg')
                )
            else:
                result = self.analyzer.analyze_heart_sound(
                    self.uploaded_file['content']
                )

            # Display results
            self.display_results(result)

    def display_results(self, result):
        """Display diagnostic results"""
        with self.analysis_output:
            clear_output()

            if result['status'] == 'error':
                display(HTML(f"""
                <div style='color:red; border-left:4px solid red; padding:10px; margin:10px 0;'>
                    <h4>Error</h4>
                    <p>{result.get('error', 'Unknown error')}</p>
                    <p><b>Recommendation:</b> {result.get('recommendation', '')}</p>
                </div>
                """))
                return

            condition = result.get('condition', {})
            color = condition.get('color', '#1a73e8')

            if self.current_mode == 'eye':
                display(HTML(f"""
                <div style='border:2px solid {color}; border-radius:8px; padding:20px; margin:15px 0;'>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                        <h2 style="color:{color}; margin-top:0;">
                            {condition.get('icon', '👁️')} {result.get('diagnosis', 'Unknown')}
                        </h2>
                        <div style="background:{color}20; color:{color}; padding:5px 15px; border-radius:20px; font-weight:bold;">
                            Confidence: {result.get('confidence', 'N/A')}
                        </div>
                    </div>
                    <div style='background:#f5f5f5; padding:15px; border-radius:5px;'>
                        <p><b>Type:</b> {condition.get('type', 'Unknown')}</p>
                        <p><b>Urgency:</b> {condition.get('urgency', 'Consult specialist')}</p>
                    </div>
                </div>
                """))
            else:
                # For heart sound analysis
                if 'visualization' in result:
                    buf = io.BytesIO()
                    result['visualization'].savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    vis_html = f"""
                    <div style="margin:15px 0; border:1px solid #eee; padding:10px; border-radius:5px;">
                        <h4 style="color:#1a73e8; margin-top:0;">Heart Sound Analysis</h4>
                        <img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
                             style="max-width:100%; border-radius:3px;"/>
                    </div>
                    """
                else:
                    vis_html = ""

                display(HTML(f"""
                <div style='border:2px solid {color}; border-radius:8px; padding:20px; margin:15px 0;'>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                        <h2 style="color:{color}; margin-top:0;">
                            {condition.get('icon', '❤️')} {result.get('diagnosis', 'Unknown')}
                        </h2>
                        <div style="background:{color}20; color:{color}; padding:5px 15px; border-radius:20px; font-weight:bold;">
                            Confidence: {result.get('confidence', 'N/A')}
                        </div>
                    </div>
                    <div style='background:#f5f5f5; padding:15px; border-radius:5px; margin-bottom:15px;'>
                        <p><b>Description:</b> {condition.get('description', 'Not specified')}</p>
                        <p><b>Urgency:</b> {condition.get('urgency', 'Consult specialist')}</p>
                    </div>
                    {vis_html}
                </div>
                """))

    def clear_analysis(self, btn):
        """Clear current analysis"""
        self.uploaded_file = None
        self.analyze_btn.disabled = True

        with self.content_display:
            clear_output()
            if self.current_mode == 'eye':
                display(HTML("""
                <p style='text-align:center; color:#666;'>
                    Upload a clear eye image or use camera for analysis<br>
                    <small>Close-up of the eye with good lighting works best</small>
                </p>
                """))
            else:
                display(HTML("""
                <p style='text-align:center; color:#666;'>
                    Upload or record heart sound for analysis<br>
                    <small>Record in quiet environment for best results</small>
                </p>
                """))

        with self.analysis_output:
            clear_output()
            display(HTML("<p style='color:#4CAF50;'>Ready for new analysis</p>"))

if __name__ == "__main__":
    print("🩺 Starting Multi-Modal Health Diagnosis System...")
    dashboard = HealthDiagnosisDashboard()
