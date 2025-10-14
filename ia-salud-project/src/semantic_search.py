import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json
from sklearn.neighbors import NearestNeighbors
from src.data_pipeline_cnn import preprocess_image_for_cnn 

# --- Rutas de Archivos ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
DESCRIPTIONS_FILE = os.path.join(DATA_DIR, 'case_descriptions.json')
EXTRACTOR_PATH = os.path.join(MODELS_DIR, 'cnn_feature_extractor.h5')

class SemanticSearch:
    """Motor para buscar el caso de entrenamiento más similar a la imagen del usuario usando Scikit-learn."""
    
    def __init__(self):
        self.extractor = self._load_extractor()
        self.features_db, self.metadata_db = self._load_or_build_features_db()
        self.nn_model = self._build_nn_model()
        self.is_ready = self.extractor is not None and self.features_db is not None and self.nn_model is not None

    def _load_extractor(self):
        if os.path.exists(EXTRACTOR_PATH):
            try:
                return keras.models.load_model(EXTRACTOR_PATH, compile=False) 
            except Exception as e:
                print(f"ERROR: No se pudo cargar el extractor de features: {e}")
                return None
        print("ERROR: Archivo cnn_feature_extractor.h5 no encontrado. ¡Debes generarlo!")
        return None

    def _load_or_build_features_db(self):
        if not os.path.exists(DESCRIPTIONS_FILE):
            print(f"ERROR: Falta el archivo de descripciones: {DESCRIPTIONS_FILE}")
            return None, None
        
        if not self.extractor:
             print("ERROR: Extractor no disponible para construir la Base de Datos.")
             return None, None

        try:
            with open(DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
                descriptions = json.load(f)
        except json.JSONDecodeError:
            print("ERROR: El archivo case_descriptions.json no es un JSON válido.")
            return None, None

        metadata_db = []
        feature_list = []
        
        print("INFO: Construyendo Features DB (por única vez)...")
        for desc in descriptions:
            class_folder = desc['class'] 
            full_path = os.path.join(DATA_DIR, 'imagenes_dicom', class_folder, desc['filename'])
            
            processed_tensor = preprocess_image_for_cnn(full_path)
            
            if processed_tensor is not None:
                try:
                    input_tensor = np.expand_dims(processed_tensor, axis=0)
                    feature = self.extractor.predict(input_tensor, verbose=0)
                    
                    # **CORRECCIÓN 1:** Remodelar el feature a 2D (1, N) para sklearn
                    feature_reshaped = feature.reshape(1, -1) 
                    feature_list.append(feature_reshaped)
                    
                    metadata_db.append({
                        'filename': desc['filename'],
                        'class': desc['class'],
                        'description': desc['description'],
                    })
                except Exception as e:
                    print(f"ERROR: Falló la extracción de features para {desc['filename']}: {e}")

        if feature_list:
            # Apilamos la lista de arrays (1, N) en un solo array (M, N)
            return np.vstack(feature_list), metadata_db 
        else:
            print("ERROR FATAL: Cero features extraídos. Verifica el JSON y las imágenes.")
            return None, None

    def _build_nn_model(self):
        if self.features_db is not None:
            nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn.fit(self.features_db)
            return nn
        return None

    def extract_features(self, processed_tensor):
        if not self.extractor:
            return None
        input_tensor = np.expand_dims(processed_tensor, axis=0) 
        feature = self.extractor.predict(input_tensor, verbose=0)
        
        # **CORRECCIÓN 2:** Remodelar el feature de usuario a 2D (1, N) para sklearn
        return feature.reshape(1, -1) 

    def find_most_similar(self, user_feature):
        if not self.is_ready or user_feature is None:
            return None

        # user_feature YA DEBE SER (1, N) gracias a la corrección en extract_features
        distances, indices = self.nn_model.kneighbors(user_feature)
        
        # Los resultados de sklearn son mucho más limpios para la extracción
        most_similar_index = int(indices[0][0].item())
        min_distance = float(distances[0][0].item()) 

        similar_case = self.metadata_db[most_similar_index]

        return {
            'distance': min_distance,
            'class': similar_case['class'],
            'description': similar_case['description'],
            'filename': similar_case['filename']
        }