import streamlit as st
import os
import sys
import numpy as np # <-- Asegúrate de que NumPy esté importado para np.exp()
import json

# Añadir el directorio 'src' al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importar funciones clave 
from src.model_utils import load_cnn_model, predict_cnn_image
from src.data_pipeline_cnn import preprocess_image_for_cnn
from src.semantic_search import SemanticSearch 

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cnn_diagnostico.h5')

# --- Configuración de la Interfaz ---
st.set_page_config(page_title="Asistente IA Médico", page_icon="🩺", layout="wide")
st.title("🩺 Asistente IA Médico Pedagógico (Final)")
st.markdown("---")

# 1. Cargar el Modelo y el Motor de Búsqueda
cnn_model = None
search_engine = None

@st.cache_resource
def load_resources():
    cnn = load_cnn_model(MODEL_PATH)
    engine = SemanticSearch()
    return cnn, engine

if os.path.exists(MODEL_PATH):
    try:
        cnn_model, search_engine = load_resources()
        
        if cnn_model and search_engine and search_engine.is_ready:
            st.sidebar.success("Modelo CNN y Buscador Pedagógico (Sklearn) cargados y listos.")
        else:
             st.sidebar.error("Error al inicializar el Buscador de Casos. Verifica los archivos.")
    except Exception as e:
        st.sidebar.error(f"Error crítico cargando componentes: {e}")
else:
    st.sidebar.warning("Aún no se ha encontrado el modelo. ¡Ejecuta 'src/train_cnn.py' primero!")


# --- Lógica de la Interfaz ---
st.header("Análisis de Imagen (Diagnóstico y Similitud)")
st.write("Sube una imagen para clasificación (CNN) y comparación con el caso más similar de la base de datos.")

uploaded_file = st.file_uploader(
    "Selecciona la imagen médica (.png, .jpg, .jpeg, o .dcm)", 
    type=['dcm', 'png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    st.image(uploaded_file, caption='Imagen cargada por el usuario', width='stretch') 
    
    if cnn_model is not None and search_engine is not None and search_engine.is_ready:
        if st.button("Obtener Análisis"):
            with st.spinner('Analizando y comparando la imagen...'):
                try:
                    image_bytes = uploaded_file.read()
                    processed_tensor = preprocess_image_for_cnn(image_bytes, from_bytes=True)
                    
                    if processed_tensor is not None:
                        # 1. CLASIFICACIÓN CNN
                        prediction_class, confidence = predict_cnn_image(cnn_model, processed_tensor)
                        
                        st.subheader("Resultado del Análisis")
                        
                        # Destacamos el resultado principal de la CNN
                        if prediction_class == "PATOLOGIA":
                            st.error(f"🚨 **Diagnóstico de la IA (CNN): {prediction_class}** (Confianza: {confidence:.2f}%)")
                        else:
                            st.success(f"✅ **Diagnóstico de la IA (CNN): {prediction_class}** (Confianza: {confidence:.2f}%)")
                            
                        st.markdown("---")
                        
                        # 2. BÚSQUEDA SEMÁNTICA (PEDAGÓGICA)
                        with st.spinner('Buscando el caso más similar en la base de datos...'):
                            user_feature = search_engine.extract_features(processed_tensor)
                            similar_case = search_engine.find_most_similar(user_feature)
                        
                        if similar_case:
                            
                            distance = similar_case['distance'] 
                            
                            # **NUEVA FÓRMULA DE SIMILITUD EXPONENCIAL**
                            # Esto evitará el 0.00% y manejará distancias grandes
                            decay_factor = 0.15 
                            similarity_score = np.exp(-decay_factor * distance) * 100

                            # Limitamos el puntaje máximo a 100.0 (no es estrictamente necesario, pero es seguro)
                            if similarity_score > 100:
                                similarity_score = 100.0
                            
                            st.markdown("##### 📚 **Comparación con Caso Pedagógico (Similitud)**")
                            st.info(f"El caso más similar en nuestra base de datos es de la clase **{similar_case['class']}**.")
                            st.metric(
                                label="Porcentaje de Parecido con el Caso Pedagógico",
                                value=f"{similarity_score:.2f}%",
                                delta=f"Distancia Euclídea: {distance:.4f}",
                                delta_color="off" 
                            )
                            st.markdown(f"**Caso de Referencia:** {similar_case['filename']}")
                            st.markdown(f"**Descripción Clínica:** {similar_case['description']}")

                        else:
                            st.warning("No se encontró una referencia clínica específica. Verifica que el Extractor de Features haya sido generado.")
                            
                    else:
                        st.error("No se pudo preprocesar la imagen. Verifica el formato o el contenido.")

                except Exception as e:
                    st.error(f"Ocurrió un error durante la predicción: {e}")
    else:
        st.warning("El modelo o el motor de búsqueda no están listos. Verifica la carga de los archivos.")

st.markdown("---")
st.caption("Sistema de diagnóstico y comparación pedagógica (ML Supervisado).")