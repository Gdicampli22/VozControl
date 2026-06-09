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
                <p style="margin-top: 40px; color: #9CA3AF; font-size: 1.2rem;">Iniciando Motor V3 (Directorio Dinámico Automatizado)...</p>
            </div>
        """, unsafe_allow_html=True)
    time.sleep(4)
    st.session_state.splash_shown = True
    st.rerun()

# --- ESTILOS PERSONALIZADOS ---
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

# --- CABECERA MEJORADA CON LOGO PROFESIONAL Y ACCESIBILIDAD ---
col_logo, col_text = st.columns([1, 5])
with col_logo:
    ruta_logo = "./imagenes/logo1.png"
    if os.path.exists(ruta_logo):
        st.image(ruta_logo, width=240)
    else:
        st.warning("⚠️ Mover logo1.png a /img")

with col_text:
    st.markdown('<div class="main-title">🎙️ Módulo de Habla — VozControl+</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Prototipo Avanzado de Clasificación de Comandos de Voz | Solución de Accesibilidad para Limitaciones Motrices | <b>Castiel Analytics</b></div>', unsafe_allow_html=True)
    st.markdown('<div class="version-badge">✅ VERSIÓN 3.5 (Inclusiva & Automatizada)</div>', unsafe_allow_html=True)

TARGET_SR = 16000
TARGET_LEN = 16000 

# --- FUNCIÓN DE EXTRACCIÓN (MANTENIDA ÍNTEGRA) ---
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

# --- MENÚ LATERAL Y CONFIGURACIÓN ---
st.sidebar.header("⚙️ Configuración del Sistema")
COMANDOS_POR_MODULO = {
    "Navegacion": ["Bajar", "Cerrar", "Menu", "Subir"],
    "Multimedia": ["Anterior", "Pausa", "Play", "Siguiente"],
    "Accesibilidad": ["Click", "Doble", "Seleccionar"],
    "Web": ["Cerrar_Pestaña", "Explorar", "Nueva_Pestaña", "Recargar"]
}
modulo_seleccionado = st.sidebar.selectbox("1. Selecciona el Módulo", list(COMANDOS_POR_MODULO.keys()))
gain_value = st.sidebar.slider("2. Amplificación de entrada (×)", 0.5, 4.0, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 Comandos disponibles para **{modulo_seleccionado}**:")
for cmd in COMANDOS_POR_MODULO[modulo_seleccionado]:
    st.sidebar.markdown(f"• <span class='sidebar-cmd'>{cmd}</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# --- LÓGICA AUTOMÁTICA DE CARGA DESDE DIRECTORIO ---
DIRECTORIO_MODELOS = "./modelos"

modelo_entrenado = None
scaler = None
label_encoder = None
feature_columns = None
features_sel = None

st.sidebar.markdown("### 📂 Estado de los Artefactos")

if not os.path.exists(DIRECTORIO_MODELOS):
    st.sidebar.error(f"⚠️ No existe la carpeta '{DIRECTORIO_MODELOS}' en la raíz del proyecto. Créala para usar la carga automática.")
else:
    archivos_carpeta = os.listdir(DIRECTORIO_MODELOS)
    modulo_key = modulo_seleccionado.lower()
    
    archivos_cargados_log = []
    
    # 1. Cargar archivo global
    file_global = next((f for f in archivos_carpeta if 'feature_names' in f.lower() or 'names' in f.lower()), None)
    if file_global:
        feature_columns = joblib.load(os.path.join(DIRECTORIO_MODELOS, file_global))
        archivos_cargados_log.append("Columnas Totales 🌐")
        
    # 2. Cargar archivos del módulo específico filtrando por nombre
    for file in archivos_carpeta:
        nombre_min = file.lower()
        path_completo = os.path.join(DIRECTORIO_MODELOS, file)
        
        if modulo_key in nombre_min:
            if 'model' in nombre_min or nombre_min.endswith('.pkl'):
                modelo_entrenado = joblib.load(path_completo)
                archivos_cargados_log.append("Modelo 🧠")
            elif 'scaler' in nombre_min:
                scaler = joblib.load(path_completo)
                archivos_cargados_log.append("Scaler ⚖️")
            elif 'le' in nombre_min or 'label' in nombre_min:
                label_encoder = joblib.load(path_completo)
                archivos_cargados_log.append("LabelEncoder 🔤")
            elif 'features_' in nombre_min:
                features_sel = joblib.load(path_completo)
                archivos_cargados_log.append("Features Seleccionadas 🎯")

    if modelo_entrenado and scaler and feature_columns and features_sel and label_encoder:
        st.sidebar.success(f"🚀 **Motor de IA listo para {modulo_seleccionado}!**")
        st.sidebar.caption(f"Componentes activos: {', '.join(archivos_cargados_log)}")
    else:
        st.sidebar.warning(f"⚠️ Esperando archivos completos en /modelos para '{modulo_seleccionado}'.")

# --- FLUJO PRINCIPAL Y PREDICCIÓN ---
audio_file = st.audio_input("Grabar o subir comando de voz:")

if audio_file is not None:
    if modelo_entrenado is None:
        st.error(f"⚠️ No hay un modelo cargado automáticamente para el módulo '{modulo_seleccionado}'. Revisa la carpeta /modelos.")
    else:
        if st.button("🚀 Procesar y Predecir", type="primary"):
            with st.spinner("Analizando espectro acústico y evaluando features..."):
                try:
                    # 1. Carga de audio y DSP
                    y, _ = librosa.load(audio_file, sr=TARGET_SR)
                    y = librosa.effects.trim(y * gain_value, top_db=25)[0]
                    y_fixed = librosa.util.fix_length(y, size=TARGET_LEN)
                    
                    # 2. Generación de Gráficos (Waveform y Espectrograma)
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].plot(np.linspace(0, 1, len(y_fixed)), y_fixed, color='#2563EB', linewidth=0.7)
                    axes[0].set_title('Waveform (Señal Limpia)')
                    
                    S = librosa.feature.melspectrogram(y=y_fixed, sr=TARGET_SR, n_mels=128)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=TARGET_SR, ax=axes[1], cmap='magma')
                    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
                    axes[1].set_title('Espectrograma de Mel')
                    plt.tight_layout()
                    
                    # 3. Extracción de características
                    features_raw = extract_features_v3(y_fixed, TARGET_SR)
                    
                    # 4. Alineación de Columnas mediante DataFrame
                    if feature_columns is not None and features_sel is not None:
                        features_df = pd.DataFrame([features_raw], columns=feature_columns)
                        df_filtrado = features_df[features_sel]
                        vector_features = df_filtrado.values
                    else:
                        vector_features = features_raw.reshape(1, -1)
                        n_esp = getattr(modelo_entrenado, "n_features_in_", 60)
                        if vector_features.shape[1] != n_esp:
                            if vector_features.shape[1] > n_esp:
                                vector_features = vector_features[:, :n_esp]
                            else:
                                vector_features = np.pad(vector_features, ((0,0), (0, n_esp - vector_features.shape[1])))

                    # 5. Escalado
                    if scaler is not None:
                        vector_features = scaler.transform(vector_features)
                    
                    # 6. Predicción Numérica
                    prediccion_numerica = modelo_entrenado.predict(vector_features)
                    
                    # 7. Traducción segura con LabelEncoder
                    if label_encoder is not None:
                        pred_matriz = np.array(prediccion_numerica).reshape(-1, 1)
                        comando_predicho = label_encoder.inverse_transform(pred_matriz.flatten())[0]
                        clases_disponibles = label_encoder.classes_
                    else:
                        lista_ordenada = sorted(COMANDOS_POR_MODULO[modulo_seleccionado])
                        indice = int(prediccion_numerica[0])
                        comando_predicho = lista_ordenada[indice % len(lista_ordenada)]
                        clases_disponibles = lista_ordenada
                    
                    # 8. Obtención de Probabilidades
                    try:
                        probs_raw = modelo_entrenado.predict_proba(vector_features)[0]
                        confianza_real = np.max(probs_raw) * 100
                    except AttributeError:
                        probs_raw = np.zeros(len(clases_disponibles))
                        confianza_real = 100.0 
                    
                    # 9. Mostrar Resultados Visuales
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
                    
                    # 10. Tabla de probabilidades por clase dinámicas
                    if hasattr(modelo_entrenado, "predict_proba"):
                        probs_sorted = sorted(zip(clases_disponibles, probs_raw * 100), key=lambda x: x[1], reverse=True)
                        st.markdown("**Probabilidades por clase:**")
                        prob_cols = st.columns(len(clases_disponibles))
                        for i, (clase, prob) in enumerate(probs_sorted):
                            with prob_cols[i]:
                                color = "#10B981" if clase == comando_predicho else "#6B7280"
                                st.markdown(f"""
                                <div style="text-align:center; padding:8px; border-radius:8px;
                                            border: 2px solid {color};">
                                    <b style="color:{color}">{clase}</b><br>
                                    <span style="font-size:1.3rem; color:{color}">{prob:.1f}%</span>
                                </div>""", unsafe_allow_html=True)
                    
                    st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error técnico durante el procesamiento: {e}")
                    import traceback
                    st.code(traceback.format_exc())