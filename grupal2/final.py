import os
from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Función para cargar los espectrogramas
def load_spectrograms(data_path, label):
    spectrograms = []
    labels = []
    for img_file in Path(data_path).glob("*.png"):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        spectrograms.append(img)
        labels.append(label)
    return spectrograms, labels

# Función para crear el modelo CNN
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Función para entrenar y guardar el modelo si no existe, o cargarlo si ya existe
def train_or_load_model(model_filename, X_train, y_train, X_test, y_test):
    if not os.path.exists(model_filename):
        print("Entrenando un nuevo modelo...")
        model = build_model()
        history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
        model.save(model_filename)
        print("Modelo entrenado y guardado correctamente.")
    else:
        print("Cargando modelo existente...")
        model = load_model(model_filename)
    return model

# Cargar los espectrogramas
output_path_male = Path("Male/")
output_path_female = Path("Female/")

male_spectrograms, male_labels = load_spectrograms(output_path_male, 0)
female_spectrograms, female_labels = load_spectrograms(output_path_female, 1)

X = np.array(male_spectrograms + female_spectrograms)
y = np.array(male_labels + female_labels)

# Normalizar los datos
X = X / 255.0
X = X.reshape(-1, 128, 128, 1)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el nombre del archivo del modelo
model_filename = "gender_classification_model.h5"

# Entrenar o cargar el modelo
model = train_or_load_model(model_filename, X_train, y_train, X_test, y_test)

# Función para grabar audio desde el micrófono
def record_audio(duration):
    print("Grabando audio...")
    samples = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float64')
    sd.wait()
    return samples.flatten(), 44100

# Función para crear espectrogramas a partir de grabaciones de micrófono
def create_fold_spectrograms_from_microphone(fold, duration=5.0):
    spectrogram_path = Path("Test/")
    print(f"Procesando fold {fold}")
    os.makedirs(spectrogram_path / f"fold{fold}", exist_ok=True)

    samples, sample_rate = record_audio(duration=duration)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    # Generar espectrograma utilizando Librosa
    S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
    filename = spectrogram_path / f"fold{fold}" / f"microphone_input.png"
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig)

# Crear un espectrograma a partir de una nueva grabación
create_fold_spectrograms_from_microphone(fold=1, duration=5.0)
new_spectrogram_path = Path("Test/fold1/microphone_input.png")

# Función para predecir el género a partir de un espectrograma
def predict_gender(spectrogram_path, model):
    img = cv2.imread(str(spectrogram_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)
    prediction = model.predict(img)
    return "Male" if prediction < 0.5 else "Female"

# Predecir el género con el modelo actual
gender = predict_gender(new_spectrogram_path, model)
print(f"Género predicho con el modelo actual: {gender}")

