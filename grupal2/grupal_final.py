import os
import numpy as np
import cv2
import sounddevice as sd
from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pathlib import Path
import threading

# Importaciones de PyQt6
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont

# Función para grabar audio desde el micrófono y guardar en archivo
def record_audio_and_save(duration, output_file):
    print("Recording audio...")
    samples = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float64')
    sd.wait()

    # Guardar los datos de audio en un archivo WAV usando scipy
    wavfile.write(output_file, 44100, (samples * (2**15)).astype(np.int16))  # Escalar y convertir a int16

    return samples.flatten(), 44100

# Función para crear espectrogramas a partir de grabaciones de micrófono
def create_spectrogram_from_audio(samples, sample_rate, output_spectrogram):
    S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
    fig = plt.figure(figsize=[4, 4])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig(output_spectrogram, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Función para predecir el género a partir de un espectrograma
def predict_gender(spectrogram_path, model):
    img = cv2.imread(str(spectrogram_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)
    prediction = model.predict(img)
    return "Masculino" if prediction < 0.5 else "Femenino"

# Clase principal de la aplicación PyQt6
class SpectrogramApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Identificador de Genero Mediante Audio")
        self.setGeometry(100, 100, 600, 600)  # Tamaño de la ventana principal

        # Etiqueta para mostrar el estado de grabación y la predicción
        self.status_label = QLabel(self)
        self.status_label.setGeometry(50, 50, 500, 100)  # Tamaño y posición de la etiqueta
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Centrar texto
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setText("Grabando audio. Habla ahora...")

        # Etiqueta para mostrar el espectrograma
        self.spectrogram_label = QLabel(self)
        self.spectrogram_label.setGeometry(50, 150, 500, 400)  # Tamaño y posición de la etiqueta

        # Cargar el modelo guardado
        self.model_path = "gender_classification_model.h5"
        self.model = load_model(self.model_path)

        # Iniciar el proceso de grabación y mostrar el estado en la ventana
        self.output_audio_file = "audio_recording.wav"
        self.duration = 5.0  # Duración de la grabación en segundos

        # Ejecutar la grabación en un hilo separado
        threading.Thread(target=self.record_audio_and_display).start()

    def record_audio_and_display(self):
        # Grabar audio desde el micrófono y guardar en archivo
        self.samples, self.sample_rate = record_audio_and_save(self.duration, self.output_audio_file)

        # Crear espectrograma a partir de la grabación de micrófono
        self.output_spectrogram = "microphone_input.png"
        create_spectrogram_from_audio(self.samples, self.sample_rate, self.output_spectrogram)

        # Mostrar el espectrograma y la predicción en la ventana
        self.show_spectrogram_and_prediction()

    def show_spectrogram_and_prediction(self):
        # Mostrar espectrograma
        pixmap = QPixmap(self.output_spectrogram)
        self.spectrogram_label.setPixmap(pixmap)
        self.spectrogram_label.setScaledContents(True)

        # Predecir el género con el modelo cargado
        new_spectrogram_path = Path(self.output_spectrogram)
        gender = predict_gender(new_spectrogram_path, self.model)

        # Mostrar la predicción de género en la ventana
        self.status_label.setText(f"Predicción de género: {gender}")

# Función principal para ejecutar la aplicación
def main():
    app = QApplication([])
    window = SpectrogramApp()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
