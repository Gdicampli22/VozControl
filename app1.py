import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="VozControl+ | Pro Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PANTALLA EMERGENTE (SPLASH SCREEN) ---
if 'splash_shown' not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    splash_container = st.empty()
    with splash_container.container():
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 85vh; text-align: center; animation: fadeIn 1s ease-in-out;">
                <h3 style="color: #60A5FA; font-weight: 300; letter-spacing: 3px; text-transform: uppercase;">Proyecto ABP</h3>
                <h1 style="font-size: 5rem; margin: 10px 0; color: #2563EB; font-weight: 900; letter-spacing: 5px;">ISPC</h1>
                <h2 style="font-size: 2rem; margin-bottom: 30px; color: #E5E7EB; font-weight: 400;">Técnicas de Procesamiento del Habla</h2>
                <div style="width: 100px; height: 4px; background-color: #10B981; margin: 20px 0;"></div>
                <h1 style="font-size: 3.5rem; margin-top: 20px; color: #10B981; font-family: 'Courier New', Courier, monospace; text-shadow: 0px 0px 10px rgba(16, 185, 129, 0.4);">
                    <span style="color: #fff;">{</span> Castiel Analytics <span style="color: #fff;">}</span>
                </h1>
                <p style="margin-top: 40px; color: #9CA3AF; font-size: 1.2rem;">Iniciando Dashboard MLOps V4.0...</p>
            </div>
        """, unsafe_allow_html=True)
    time.sleep(3.5)
    st.session_state.splash_shown = True
    st.rerun()

# --- ESTILOS CSS PROFESIONALES ---
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1rem; color: #64748B; margin-bottom: 1rem; font-weight: 400; }
    .badge-pro { background: linear-gradient(90deg, #10B981, #059669); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: bold; letter-spacing: 1px; display: inline-block; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2); }
    .metric-value { font-size: 2.5rem; font-weight: 900; color: #1E40AF; line-height: 1.2; }
    .metric-label { font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    .prob-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px; text-align: center; transition: all 0.2s ease;}
    .prob-box:hover { transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- CABECERA SUPERIOR ---
col_logo, col_text = st.columns([1, 6])
with col_logo:
    ruta_logo = "./imagenes/logo1.png"
    if os.path.exists(ruta_logo):
        st.image(ruta_logo, width=340)
    else:
        st.warning("⚠️ Logo no encontrado en /imagenes")

with col_text:
    st.markdown('<div class="main-title">🎙️ Interfaz de Inferencia: VozControl+</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Clasificación Acústica Avanzada | Solución de Accesibilidad Motriz | <b>Castiel Analytics</b></div>', unsafe_allow_html=True)
    st.markdown('<div class="badge-pro">VERSIÓN 1.5 - VOZCONTROL+</div>', unsafe_allow_html=True)

TARGET_SR = 16000
TARGET_LEN = 16000 
UMBRAL_SILENCIO = 0.015 

# --- FUNCIÓN DE EXTRACCIÓN (INTACTA) ---
def extract_features_v3(audio, sr):
    features = []
    n_samples = len(audio)
    n_fft = min(n_samples, 1024)
    hop_length = n_fft // 4
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=26, n_fft=n_fft, hop_length=hop_length)
    features.extend(np.mean(mfccs, axis=1)); features.extend(np.std(mfccs, axis=1))
    delta_mfcc = librosa.feature.delta(mfccs)
    features.extend(np.mean(delta_mfcc, axis=1))
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.extend(np.mean(chroma, axis=1))
    n_mels = min(128, n_fft // 2 + 1)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)
    features.extend(np.mean(mel_db, axis=1))
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
    features.append(np.mean(zcr)); features.append(np.std(zcr))
    try:
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features.extend(np.mean(contrast, axis=1))
    except Exception: features.extend([0.0] * 7)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.append(np.mean(centroid)); features.append(np.std(centroid))
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.append(np.mean(rolloff)); features.append(np.std(rolloff))
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    features.append(np.mean(flatness))
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.append(np.mean(bandwidth))
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)
    features.extend([np.mean(rms), np.std(rms), np.max(rms)])
    return np.array(features, dtype=np.float32)

# --- BARRA LATERAL (AJUSTES Y ARTEFACTOS) ---
st.sidebar.markdown("### ⚙️ Panel de Control")
COMANDOS_POR_MODULO = {
    "Navegacion": ["Bajar", "Cerrar", "Menu", "Subir"],
    "Multimedia": ["Anterior", "Pausa", "Play", "Siguiente"],
    "Accesibilidad": ["Click", "Doble", "Seleccionar"],
    "Web": ["Cerrar_Pestaña", "Explorar", "Nueva_Pestaña", "Recargar"]
}
modulo_seleccionado = st.sidebar.selectbox("Módulo Activo", list(COMANDOS_POR_MODULO.keys()))

with st.sidebar.expander("🛠️ Ajustes de DSP", expanded=False):
    gain_value = st.slider("Ganancia Pre-Procesamiento (×)", 0.5, 4.0, 1.0, 0.5)
    st.caption("Aumenta la ganancia si el micrófono es muy sensible o capta bajo volumen.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Motor de Inferencia")

DIRECTORIO_MODELOS = "./modelos"
modelo_entrenado, scaler, label_encoder, feature_columns, features_sel = None, None, None, None, None

if not os.path.exists(DIRECTORIO_MODELOS):
    st.sidebar.error(f"Falta directorio '{DIRECTORIO_MODELOS}'.")
else:
    archivos_carpeta = os.listdir(DIRECTORIO_MODELOS)
    modulo_key = modulo_seleccionado.lower()
    
    file_global = next((f for f in archivos_carpeta if 'feature_names' in f.lower() or 'names' in f.lower()), None)
    if file_global: feature_columns = joblib.load(os.path.join(DIRECTORIO_MODELOS, file_global))
        
    for file in archivos_carpeta:
        nombre_min = file.lower()
        path_completo = os.path.join(DIRECTORIO_MODELOS, file)
        
        if modulo_key in nombre_min:
            if 'model' in nombre_min or nombre_min.endswith('.pkl'): modelo_entrenado = joblib.load(path_completo)
            elif 'scaler' in nombre_min: scaler = joblib.load(path_completo)
            elif 'le' in nombre_min or 'label' in nombre_min: label_encoder = joblib.load(path_completo)
            elif 'features_' in nombre_min: features_sel = joblib.load(path_completo)

    if modelo_entrenado and scaler and feature_columns and features_sel and label_encoder:
        st.sidebar.success("🟢 Sistema Online y Cargado")
        st.sidebar.markdown(f"**Diccionario del módulo:**")
        for cmd in COMANDOS_POR_MODULO[modulo_seleccionado]:
            st.sidebar.markdown(f"➤ `{cmd}`")
    else:
        st.sidebar.warning("🟡 Faltan artefactos de IA para este módulo.")

# --- CUERPO PRINCIPAL DEL DASHBOARD ---

# Contenedor 1: Captura de Audio
with st.container(border=True):
    st.markdown("#### 1️⃣ Origen de la Señal Acústica")
    tab_mic, tab_upload = st.tabs(["🎙️ Capturar Micrófono", "📁 Cargar Archivo WAV/MP3"])
    audio_file = None
    
    with tab_mic:
        st.caption("Asegúrate de hablar claro. El sistema cuenta con Detección de Actividad de Voz (VAD).")
        audio_mic = st.audio_input("Grabar comando:")
        if audio_mic: audio_file = audio_mic

    with tab_upload:
        audio_upload = st.file_uploader("Arrastra un archivo de audio", type=['wav', 'mp3', 'ogg'], label_visibility="collapsed")
        if audio_upload: 
            st.audio(audio_upload)
            audio_file = audio_upload

# Lógica de Procesamiento
if audio_file is not None:
    if modelo_entrenado is None:
        st.error("No se puede predecir: Faltan archivos del modelo.")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ Ejecutar Predicción de Comando", type="primary", use_container_width=True):
            with st.spinner("Procesando pipeline de datos..."):
                try:
                    # 1. DSP y VAD
                    y, _ = librosa.load(audio_file, sr=TARGET_SR)
                    y_amplificado = y * gain_value
                    energia_rms = np.max(librosa.feature.rms(y=y_amplificado))
                    
                    if energia_rms < UMBRAL_SILENCIO:
                        st.error(f"🛑 **Interrupción VAD:** Señal demasiado débil (RMS: {energia_rms:.4f}). Se detectó silencio o solo ruido de fondo.")
                    else:
                        y_trimmed = librosa.effects.trim(y_amplificado, top_db=25)[0]
                        y_fixed = librosa.util.fix_length(y_trimmed, size=TARGET_LEN)
                        
                        # 2. Extracción
                        features_raw = extract_features_v3(y_fixed, TARGET_SR)
                        if feature_columns is not None and features_sel is not None:
                            features_df = pd.DataFrame([features_raw], columns=feature_columns)
                            vector_features = features_df[features_sel].values
                        else:
                            vector_features = features_raw.reshape(1, -1)
                            n_esp = getattr(modelo_entrenado, "n_features_in_", 60)
                            vector_features = vector_features[:, :n_esp] if vector_features.shape[1] > n_esp else np.pad(vector_features, ((0,0), (0, n_esp - vector_features.shape[1])))

                        if scaler: vector_features = scaler.transform(vector_features)
                        
                        # 3. Predicción
                        prediccion_numerica = modelo_entrenado.predict(vector_features)
                        
                        if label_encoder is not None:
                            comando_predicho = label_encoder.inverse_transform(np.array(prediccion_numerica).reshape(-1, 1).flatten())[0]
                            clases_disponibles = label_encoder.classes_
                        else:
                            lista_ordenada = sorted(COMANDOS_POR_MODULO[modulo_seleccionado])
                            comando_predicho = lista_ordenada[int(prediccion_numerica[0]) % len(lista_ordenada)]
                            clases_disponibles = lista_ordenada
                        
                        try:
                            probs_raw = modelo_entrenado.predict_proba(vector_features)[0]
                            confianza_real = np.max(probs_raw) * 100
                        except AttributeError:
                            probs_raw = np.zeros(len(clases_disponibles))
                            confianza_real = 100.0 

                        # --- CONTENEDOR 2: RESULTADOS PRINCIPALES ---
                        st.markdown("#### 2️⃣ Resultados de Clasificación")
                        with st.container(border=True):
                            col_res1, col_res2 = st.columns(2)
                            with col_res1:
                                st.markdown(f'<div class="metric-label">Comando Identificado</div><div class="metric-value" style="color: #10B981;">{comando_predicho}</div>', unsafe_allow_html=True)
                            with col_res2:
                                st.markdown(f'<div class="metric-label">Nivel de Confianza (Softmax)</div><div class="metric-value">{confianza_real:.2f}%</div>', unsafe_allow_html=True)
                            
                            st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
                            st.markdown("<div class='metric-label' style='margin-bottom:10px;'>Distribución de Probabilidades del Ensamble</div>", unsafe_allow_html=True)
                            
                            # Grilla de probabilidades limpia
                            if hasattr(modelo_entrenado, "predict_proba"):
                                probs_sorted = sorted(zip(clases_disponibles, probs_raw * 100), key=lambda x: x[1], reverse=True)
                                cols = st.columns(len(clases_disponibles))
                                for i, (clase, prob) in enumerate(probs_sorted):
                                    with cols[i]:
                                        bg_color = "#ECFDF5" if clase == comando_predicho else "#F8FAFC"
                                        border_color = "#10B981" if clase == comando_predicho else "#E2E8F0"
                                        text_color = "#065F46" if clase == comando_predicho else "#64748B"
                                        st.markdown(f"""
                                        <div class="prob-box" style="background-color: {bg_color}; border-color: {border_color};">
                                            <div style="font-size: 0.8rem; font-weight: bold; color: {text_color}; text-transform: uppercase;">{clase}</div>
                                            <div style="font-size: 1.2rem; font-weight: 800; color: {text_color};">{prob:.1f}%</div>
                                        </div>
                                        """, unsafe_allow_html=True)

                        # --- CONTENEDOR 3: ANÁLISIS TÉCNICO (OCULTO POR DEFECTO) ---
                        with st.expander("🔬 Ver Análisis Espectral y Señal DSP", expanded=False):
                            st.caption(f"Análisis generado tras VAD. Energía RMS detectada: {energia_rms:.4f}")
                            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                            axes[0].plot(np.linspace(0, 1, len(y_fixed)), y_fixed, color='#2563EB', linewidth=0.5)
                            axes[0].set_title('Waveform (Normalizada)')
                            axes[0].axis('off')
                            
                            S = librosa.feature.melspectrogram(y=y_fixed, sr=TARGET_SR, n_mels=128)
                            S_dB = librosa.power_to_db(S, ref=np.max)
                            librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=TARGET_SR, ax=axes[1], cmap='magma')
                            axes[1].set_title('Espectrograma de Mel')
                            axes[1].axis('off')
                            
                            plt.tight_layout()
                            st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error técnico durante la inferencia: {e}")