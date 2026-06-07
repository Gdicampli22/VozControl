import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="VozControl+ | Castiel Analytics",
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
                <p style="margin-top: 40px; color: #9CA3AF; font-size: 1.2rem;">Iniciando Motor V2.7...</p>
            </div>
        """, unsafe_allow_html=True)
    time.sleep(4)
    st.session_state.splash_shown = True
    st.rerun()

# ==========================================
# A PARTIR DE AQUÍ COMIENZA LA APP PRINCIPAL
# ==========================================

# Estilos personalizados
st.markdown("""
    <style id="estilos-v2-7">
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.1rem; color: #4B5563; margin-bottom: 0.5rem; }
    .version-badge { background-color: #10B981; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold; display: inline-block; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .metric-box { background-color: #F3F4F6; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #2563EB; margin-bottom: 1rem; }
    .command-text { font-size: 1.8rem; font-weight: bold; color: #10B981; }
    .sidebar-cmd { font-size: 1.1rem; color: #10B981; font-weight: bold; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎙️ Módulo de Habla — Versión Mejorada</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prototipo Avanzado de Clasificación de Comandos de Voz (Audio DSP + Machine Learning) | <b>Castiel Analytics</b></div>', unsafe_allow_html=True)
st.markdown('<div class="version-badge">✅ VERSIÓN 2.7 (Auto-Traductor Integrado)</div>', unsafe_allow_html=True)

# Constantes del pipeline
TARGET_SR = 16000
TARGET_LEN = 16000 

# --- FUNCIÓN DE EXTRACCIÓN DE CARACTERÍSTICAS V3 ---
def extract_features_v3(audio, sr):
    features = []
    n_samples = len(audio)
    n_fft = min(n_samples, 1024)
    hop_length = n_fft // 4

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=26, n_fft=n_fft, hop_length=hop_length)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    delta_mfcc = librosa.feature.delta(mfccs)
    features.extend(np.mean(delta_mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.extend(np.mean(chroma, axis=1))

    n_mels = min(128, n_fft // 2 + 1)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)
    features.extend(np.mean(mel_db, axis=1))

    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    try:
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features.extend(np.mean(contrast, axis=1))
    except Exception:
        features.extend([0.0] * 7)

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    features.append(np.mean(flatness))

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.append(np.mean(bandwidth))

    rms = librosa.feature.rms(y=audio, hop_length=hop_length)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    features.append(np.max(rms))

    return np.array(features, dtype=np.float32)


# ==========================================
# --- MENÚ LATERAL RESTAURADO ---
# ==========================================
st.sidebar.header("⚙️ Configuración del Sistema")

# Diccionario exacto de los comandos según el PDF de tu modelo
COMANDOS_POR_MODULO = {
    "Navegacion": ["Bajar", "Cerrar", "Menu", "Subir"],
    "Multimedia": ["Anterior", "Pausa", "Play", "Siguiente"],
    "Accesibilidad": ["Click", "Doble", "Seleccionar"],
    "Web": ["Cerrar_Pestaña", "Explorar", "Nueva_Pestaña", "Recargar"]
}

# 1. Selector de Módulo
modulo_seleccionado = st.sidebar.selectbox("Selecciona el Módulo", list(COMANDOS_POR_MODULO.keys()))

# 2. Slider de Amplificación
gain_value = st.sidebar.slider("3. Amplificación de entrada (×)", 0.1, 5.0, 1.0, 0.1)

st.sidebar.markdown("---")

# 3. Lista de Comandos Visual
st.sidebar.markdown("### 📋 Comandos del Módulo")
for cmd in COMANDOS_POR_MODULO[modulo_seleccionado]:
    st.sidebar.markdown(f"• &nbsp; <span class='sidebar-cmd'>{cmd}</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")

# 4. Cargador de Modelos (Adaptado al módulo seleccionado)
st.sidebar.header("🧠 Entorno Predictivo")
st.sidebar.info(f"Sube los archivos (.joblib) del módulo **{modulo_seleccionado}** (Modelo y Scaler requeridos).")

archivos_subidos = st.sidebar.file_uploader(
    "Selecciona archivos de IA", 
    type=["joblib", "pkl"], 
    accept_multiple_files=True,
    key=f"uploader_{modulo_seleccionado}" 
)

# Variables globales para los modelos
modelo_entrenado = None
scaler = None
label_encoder = None

# Procesar archivos subidos
if archivos_subidos:
    archivos_encontrados = set()

    for file in archivos_subidos:
        temp_name = f"temp_{file.name}"
        with open(temp_name, "wb") as f:
            f.write(file.getbuffer())
        
        try:
            nombre_min = file.name.lower()
            if 'model' in nombre_min or nombre_min.endswith('.pkl'):
                modelo_entrenado = joblib.load(temp_name)
                archivos_encontrados.add('model')
            elif 'scaler' in nombre_min:
                scaler = joblib.load(temp_name)
                archivos_encontrados.add('scaler')
            elif 'le' in nombre_min or 'label' in nombre_min:
                label_encoder = joblib.load(temp_name)
                archivos_encontrados.add('le')
            elif 'features' in nombre_min:
                archivos_encontrados.add('features')
            
            os.remove(temp_name)
        except Exception as e:
            st.sidebar.error(f"Error al leer {file.name}: {e}")

    if 'model' in archivos_encontrados and 'scaler' in archivos_encontrados:
        st.sidebar.success("✅ Motor de IA listo (Modelo + Scaler).")
    elif 'model' in archivos_encontrados:
        st.sidebar.success("✅ Modelo cargado (Sin normalizar).")

# ==========================================
# --- FLUJO PRINCIPAL ---
# ==========================================
st.subheader("📥 Entrada de Audio de Voz")
col_input, col_info = st.columns([2, 1])

with col_input:
    origen_audio = st.radio("Método de entrada:", ["Grabar Micrófono", "Subir Archivo (.wav)"], horizontal=True)
    audio_file = None
    
    if origen_audio == "Grabar Micrófono":
        audio_file = st.audio_input("Graba tu comando de voz:")
    else:
        audio_file = st.file_uploader("Selecciona un archivo de audio crudo:", type=["wav"])

with col_info:
    st.info("""
    **Indicaciones de Uso:**
    1. Asegúrate de hablar claro y cerca del micrófono.
    2. El sistema aplica Trimming automático (`top_db=25`).
    3. Usa comandos correspondientes al módulo activo.
    """)

if audio_file is not None:
    if modelo_entrenado is None:
        st.error(f"⚠️ Por favor, sube el modelo para el módulo '{modulo_seleccionado}' en la barra lateral antes de procesar el audio.")
    else:
        if st.button("🚀 Procesar y Predecir", type="primary"):
            with st.spinner("Analizando espectro acústico..."):
                try:
                    # 1. Guardar y procesar audio
                    temp_filename = "temp_input.wav"
                    with open(temp_filename, "wb") as f:
                        f.write(audio_file.getbuffer())
                    
                    audio_raw, sr_raw = librosa.load(temp_filename, sr=TARGET_SR)
                    audio_raw = audio_raw * gain_value
                    
                    # 2. Limpieza
                    audio_trimmed, _ = librosa.effects.trim(audio_raw, top_db=25)
                    audio_processed = audio_trimmed / np.max(np.abs(audio_trimmed)) if np.max(np.abs(audio_trimmed)) > 0 else audio_trimmed
                    audio_fixed = librosa.util.fix_length(audio_processed, size=TARGET_LEN)
                    
                    # 3. Gráficos
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].plot(np.linspace(0, 1, len(audio_fixed)), audio_fixed, color='teal', linewidth=0.7)
                    axes[0].set_title('Waveform (Señal Limpia)')
                    
                    S = librosa.feature.melspectrogram(y=audio_fixed, sr=TARGET_SR, n_mels=128, fmax=8000)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=TARGET_SR, fmax=8000, ax=axes[1], cmap='magma')
                    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
                    axes[1].set_title('Espectrograma de Mel')
                    plt.tight_layout()
                    
                    # 4. Inferencia Real
                    vector_features = extract_features_v3(audio_fixed, TARGET_SR).reshape(1, -1)
                    
                    if scaler is not None:
                        vector_features = scaler.transform(vector_features)
                    
                    prediccion_numerica = modelo_entrenado.predict(vector_features)
                    
                    # --- NUEVA LÓGICA DE AUTO-TRADUCCIÓN ---
                    if label_encoder is not None:
                        comando_predicho = label_encoder.inverse_transform(prediccion_numerica)[0]
                    else:
                        # Fallback inteligente: ordena alfabéticamente las palabras y busca el índice exacto
                        lista_ordenada = sorted(COMANDOS_POR_MODULO[modulo_seleccionado])
                        indice = int(prediccion_numerica[0])
                        comando_predicho = lista_ordenada[indice]
                    
                    try:
                        confianza_real = np.max(modelo_entrenado.predict_proba(vector_features)) * 100
                    except AttributeError:
                        confianza_real = 100.0 
                    
                    # 5. Mostrar Resultados Visuales
                    st.markdown("---")
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <p style="margin:0; font-size:0.9rem; color:#6B7280; font-weight:bold;">COMANDO DETECTADO</p>
                            <p class="command-text">{comando_predicho}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_res2:
                        st.markdown(f"""
                        <div class="metric-box" style="border-left-color: #F59E0B;">
                            <p style="margin:0; font-size:0.9rem; color:#6B7280; font-weight:bold;">NIVEL DE CONFIANZA</p>
                            <p class="command-text" style="color:#F59E0B;">{confianza_real:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.pyplot(fig)
                    
                    if os.path.exists(temp_filename): os.remove(temp_filename)
                        
                except Exception as e:
                    st.error(f"Error técnico durante el procesamiento DSP: {e}")