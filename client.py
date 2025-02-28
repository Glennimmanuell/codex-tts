import sys
import requests
import re
import whisper
from gtts import gTTS
from langdetect import detect
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
import soundfile as sf
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class CodexWorker(QThread):
    response_signal = pyqtSignal(str, str)
    
    def __init__(self, codex_host, prompt):
        super().__init__()
        self.codex_host = codex_host
        self.prompt = prompt
    
    def run(self):
        url = f"http://{self.codex_host}/api/generate"
        payload = {"model": "deepseek-r1:14b", "prompt": self.prompt, "stream": False}
        
        try:
            response = requests.post(url, json=payload)
            response_data = response.json()
            response_text = response_data.get("response", "")
            cleaned_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            cleaned_text = re.sub(r'[/*]+', '', cleaned_text).strip()
            self.response_signal.emit(response_text, cleaned_text)
        except Exception as e:
            self.response_signal.emit(f"Error mendapatkan respon: {e}", "")

class SpeechRecognitionWorker(QThread):
    text_signal = pyqtSignal(str)
    
    def __init__(self, whisper_model):
        super().__init__()
        self.whisper_model = whisper_model
    
    def run(self):
        try:
            duration = 5
            sample_rate = 16000
            
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
            sd.wait()
            temp_audio_file = "temp_audio.wav"
            sf.write(temp_audio_file, audio_data, sample_rate)
            
            result = self.whisper_model.transcribe(temp_audio_file)
            text = result["text"].strip()
            
            if text:
                self.text_signal.emit(text)
            else:
                self.text_signal.emit("Tidak ada suara yang terdeteksi.")
        except Exception as e:
            self.text_signal.emit(f"Error saat menangkap suara: {e}")

class TextToSpeechWorker(QThread):
    
    def __init__(self, text):
        super().__init__()
        self.text = text
    
    def run(self):
        try:
            lang = detect(self.text)
            lang = 'id' if lang == 'id' else 'en'
            tts = gTTS(text=self.text, lang=lang)
            tts.save("temp_tts.mp3")
            
            audio = AudioSegment.from_mp3("temp_tts.mp3")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            samples /= np.iinfo(audio.array_type).max
            
            sd.play(samples, audio.frame_rate)
            sd.wait()
        except Exception as e:
            print(f"Error saat menghasilkan suara: {e}")

class CodexTTSApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.codex_host = "codex.petra.ac.id:11434"
        self.whisper_model = whisper.load_model("small")
    
    def initUI(self):
        self.setWindowTitle("Codex AI Voice Assistant")
        self.setGeometry(100, 100, 400, 300)
        
        self.label = QLabel("Tekan tombol untuk mulai berbicara:", self)
        self.label.setAlignment(Qt.AlignCenter)
        
        self.recordButton = QPushButton("🎤 Mulai Bicara", self)
        self.recordButton.clicked.connect(self.startListening)
        
        self.responseText = QTextEdit(self)
        self.responseText.setReadOnly(True)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.recordButton)
        layout.addWidget(self.responseText)
        
        self.setLayout(layout)
    
    def startListening(self):
        self.recordButton.setEnabled(False)
        self.label.setText("🎧 Mendengarkan...")
        self.sr_worker = SpeechRecognitionWorker(self.whisper_model)
        self.sr_worker.text_signal.connect(self.handleSpeechResult)
        self.sr_worker.start()
    
    def handleSpeechResult(self, text):
        self.label.setText("Tekan tombol untuk mulai berbicara:")
        self.recordButton.setEnabled(True)
        
        if text:
            self.responseText.setText(f"📝 Anda berkata: {text}\n\n🔄 Memproses...")
            self.getAIResponse(text)
    
    def getAIResponse(self, text):
        self.codex_worker = CodexWorker(self.codex_host, text)
        self.codex_worker.response_signal.connect(self.handleAIResponse)
        self.codex_worker.start()
    
    def handleAIResponse(self, response, cleaned_text):
        self.responseText.setText(f"💬 Respon Codex:\n{cleaned_text}")
        
        if cleaned_text:
            self.tts_worker = TextToSpeechWorker(cleaned_text)
            self.tts_worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CodexTTSApp()
    window.show()
    sys.exit(app.exec_())