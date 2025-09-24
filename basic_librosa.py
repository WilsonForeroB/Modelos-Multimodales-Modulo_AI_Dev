import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# escuchar en python
import sounddevice as sd

# Archivo de audio de ejemplo (puedes reemplazarlo por el tuyo)
AUDIO_FILE = librosa.example('trumpet')  # Descarga un ejemplo incluido en Librosa
#AUDIO_FILE = librosa.load("bongo.wav")

# 1. Carga y guardado de audio
def cargar_audio(file=AUDIO_FILE):
    y, sr = librosa.load(file)
    print(f"Audio cargado con {len(y)} muestras y frecuencia de muestreo {sr} Hz")
    return y, sr


# 2. Manipulación de señales (ej: resampling)
def resample_audio(y, sr, target_sr=16000):
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    print(f"Señal remuestreada a {target_sr} Hz")
    return y_resampled, target_sr


# 3. Transformaciones (STFT)
def calcular_stft(y):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_db, sr=22050, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.f dB")
    plt.title("Espectrograma (STFT)")
    plt.show()


# 4. Extracción de características (MFCCs)
def extraer_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print("Forma de las MFCCs:", mfccs.shape)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title("MFCCs")
    plt.show()


# 5. Análisis tiempo-frecuencia (Chroma)
def analizar_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title("Características Chroma")
    plt.show()


# 6. Detección de eventos (Onsets)
def detectar_onsets(y, sr):
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    print("Onsets detectados (segundos):", onsets)
    plt.figure(figsize=(8, 2))
    plt.plot(y, label="Señal de audio")
    for onset in librosa.time_to_samples(onsets, sr=sr):
        plt.axvline(x=onset, color="r", linestyle="--")
    plt.title("Detección de Onsets")
    plt.legend()
    plt.show()



# ===========================
# EJEMPLO DE USO LIBROSA
# ===========================
if __name__ == "__main__":
    
    y, sr = librosa.load(librosa.example('trumpet'))

    # Reproducir
    sd.play(y, sr)
    sd.wait() 

    # 2. Resamplear
    y_resampled, sr_resampled = resample_audio(y, sr)

    # 3. Transformación (STFT)
    calcular_stft(y_resampled)

    # 4. Extraer MFCCs
    extraer_mfcc(y_resampled, sr_resampled)

    # 5. Análisis Chroma
    analizar_chroma(y_resampled, sr_resampled)

    # 6. Detección de eventos
    detectar_onsets(y_resampled, sr_resampled)
