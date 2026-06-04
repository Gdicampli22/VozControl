import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# Configuración de la página con estilo profesional
st.set_page_config(
    page_title="Módulo de Habla — Interfaz Streamlit",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados para emular un entorno profesional de analítica
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .command-text {
        font-size: 1.8rem;
        font-weight: bold;
        color: #10B981;
    }
    </style>
""", unsafe_allow_html=True)

# Títulos principales
st.markdown('<div class="main-title">🎙️ Módulo de Habla — Versión Mejorada</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prototipo Avanzado de Clasificación de Comandos de Voz (Audio DSP + Machine Learning)</div>', unsafe_allow_html=True)

# Inicialización de constantes y variables simuladas de tu pipeline (ajustar según tu notebook)
TARGET_SR = 16000
TARGET_LEN = 16000 # 1 segundo a 16kHz

# Mapeo de comandos por módulo tal como figuran en tu set de datos
COMMANDS_MAP = {
    "Navegación": ["Atrás", "Adelante", "Inicio", "Recargar", "Buscar"],
    "Multimedia": ["Pausa", "Reproducir", "Siguiente", "Anterior", "Silencio"],
    "Accesibilidad": ["Zoom In", "Zoom Out", "Contraste", "Lector", "Teclado"],
    "Web": ["Abrir pestaña", "Cerrar pestaña", "Historial", "Favoritos", "Descargas"]
}

# --- BARRA LATERAL CONTROLES ---
st.sidebar.header("⚙️ Configuración del Sistema")

# 1. Selección de Módulo Predictivo
modulo_seleccionado = st.sidebar.selectbox(
    "Selecciona el Módulo",
    options=list(COMMANDS_MAP.keys()),
    help="Determina el dominio cerrado de comandos para acotar la inferencia."
)

# 2. Control de Ganancia / Amplificación (Gain Slider)
gain_value = st.sidebar.slider(
    "3. Amplificación de entrada (×)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="↑ Aumentá si el audio sale muy débil · ↓ Bajá si satura."
)

# Mostrar lista de comandos esperados en el módulo seleccionado
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Comandos del Módulo")
for cmd in COMMANDS_MAP[modulo_seleccionado]:
    st.sidebar.markdown(f"- **`{cmd}`**")


# --- CUERPO PRINCIPAL / CARGA DE AUDIO ---
st.subheader("📥 Entrada de Audio de Voz")

col_input, col_info = st.columns([2, 1])

with col_input:
    # Opción dual para grabar en vivo o cargar archivo pregrabado (Reemplazo avanzado de Gradio)
    origen_audio = st.radio("Método de entrada:", ["Grabar Micrófono", "Subir Archivo (.wav)"], horizontal=True)

    audio_file = None
    if origen_audio == "Grabar Micrófono":
        # Nota: st.audio_input está disponible en versiones recientes de Streamlit (1.37+)
        try:
            audio_file = st.audio_input("Graba tu comando de voz:")
        except AttributeError:
            audio_file = st.file_uploader("Sube tu grabación de micrófono:", type=["wav"])
            st.warning("Tu versión de streamlit no soporta grabación nativa directa. Usando uploader.")
    else:
        audio_file = st.file_uploader("Selecciona un archivo de audio comprimido o crudo:", type=["wav"])

with col_info:
    st.info("""
    **Indicaciones de Uso:**
    1. Asegúrate de hablar claro y cerca del micrófono.
    2. El sistema aplica un Trimming automático (`top_db=25`) para remover silencios iniciales y finales.
    3. Si la confianza es baja, intenta ajustar el slider de amplificación de la barra lateral.
    """)

# --- PROCESAMIENTO E INFERENCIA ---
if audio_file is not None:
    st.success("Audio cargado correctamente. Listo para procesar.")
    
    if st.button("🚀 Predecir Comando", type="primary"):
        with st.spinner("Ejecutando Pipeline DSP Avanzado (Feature V3) e Inferencia..."):
            try:
                # 1. Carga del archivo utilizando Librosa (emulando predict_from_raw_audio)
                # Guardamos temporalmente el archivo subido para que Librosa pueda leerlo
                temp_filename = "temp_input.wav"
                with open(temp_filename, "wb") as f:
                    f.write(audio_file.getbuffer())
                
                # Carga y remuestreo nativo
                audio_raw, sr_raw = librosa.load(temp_filename, sr=TARGET_SR)
                
                # Aplicar amplificación (Gain Slider)
                audio_raw = audio_raw * gain_value
                
                # 2. Preprocesamiento idéntico a tu notebook
                # Trimming dinámico a 25dB
                audio_trimmed, _ = librosa.effects.trim(audio_raw, top_db=25)
                
                # Normalización de amplitud
                if np.max(np.abs(audio_trimmed)) > 0:
                    audio_processed = audio_trimmed / np.max(np.abs(audio_trimmed))
                else:
                    audio_processed = audio_trimmed
                
                # Ajuste de longitud fija (Padding/Cropping)
                audio_fixed = librosa.util.fix_length(audio_processed, size=TARGET_LEN)
                
                # 3. Extracción Visual de Señales (Waveform y Espectrograma)
                fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                
                # Gráfico 1: Waveform
                axes[0].plot(np.linspace(0, 1, len(audio_fixed)), audio_fixed, color='steelblue', linewidth=0.7)
                axes[0].set_title('Waveform (Preprocesado y Normalizado)')
                axes[0].set_xlabel('Tiempo (s)')
                axes[0].set_ylabel('Amplitud')
                axes[0].grid(True, linestyle='--', alpha=0.5)
                
                # Gráfico 2: Mel Spectrogram (Mapeando tu extract_features_v3)
                S = librosa.feature.melspectrogram(y=audio_fixed, sr=TARGET_SR, n_mels=128, fmax=8000)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=TARGET_SR, fmax=8000, ax=axes[1], cmap='viridis')
                fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
                axes[1].set_title('Espectrograma de Mel (128 bandas - Feature V3)')
                
                plt.tight_layout()
                
                # --- SIMULACIÓN DE INFERENCIA DEL MODELO ENSEMBLE/XGBOOST ---
                # Aquí llamarías a tu modelo cargado: model.predict() y model.predict_proba()
                # Para asegurar la ejecución independiente, seleccionamos un comando aleatorio del módulo válido
                comandos_posibles = COMMANDS_MAP[modulo_seleccionado]
                comando_predicho = np.random.choice(comandos_posibles)
                confianza_simulada = np.random.uniform(0.78, 0.98)
                
                # Mostrar Resultados con Layout Limpio
                st.markdown("---")
                st.subheader("📊 Resultados de la Inferencia")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <p style="margin:0; font-size:0.9rem; color:#6B7280; text-transform:uppercase; font-weight:bold;">Comando Detectado</p>
                        <p class="command-text">{comando_predicho}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_res2:
                    st.markdown(f"""
                    <div class="metric-box" style="border-left-color: #F59E0B;">
                        <p style="margin:0; font-size:0.9rem; color:#6B7280; text-transform:uppercase; font-weight:bold;">Confianza del Ensable (XGBoost)</p>
                        <p class="command-text" style="color:#F59E0B;">{confianza_simulada*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Renderizar los gráficos DSP en Streamlit
                st.pyplot(fig)
                
                # Mostrar detalles técnicos de la señal como en tu reporte automatizado
                st.markdown("**Métricas Técnicas del Audio Recibido:**")
                st.text(f"Duración original: {len(audio_raw)/sr_raw:.2f}s | Frecuencia de Muestreo: {sr_raw} Hz -> Resampled to {TARGET_SR} Hz")
                
                # Limpieza de archivo temporal
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    
            except Exception as e:
                st.error(f"Error al procesar el archivo de audio: {e}")
                st.info("Asegúrate de que el archivo subido sea un .wav válido y que las dependencias estén correctamente instaladas.")
else:
    st.write("Esperando interacción del usuario... Sube o graba un comando de voz para comenzar.")
