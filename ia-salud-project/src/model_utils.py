import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def load_cnn_model(model_path):
    """Carga el modelo CNN entrenado desde la ruta especificada."""
    try:
        # Usamos compile=False ya que solo haremos inferencia
        model = keras.models.load_model(model_path, compile=False) 
        return model
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return None

def predict_cnn_image(model, processed_tensor):
    """
    Realiza la predicción con el modelo CNN en un tensor preprocesado.
    """
    # Añadir la dimensión del batch (1, 256, 256, 3)
    input_tensor = np.expand_dims(processed_tensor, axis=0) 
    
    # Realizar predicción
    predictions = model.predict(input_tensor, verbose=0)[0]
    
    # Asumimos que Keras ordena las clases alfabéticamente: 0=NORMAL, 1=PATOLOGIA
    class_labels = ["NORMAL", "PATOLOGIA"] 
    
    # Encontrar la clase con la probabilidad más alta
    predicted_index = int(np.argmax(predictions)) 
    
    prediction_class = class_labels[predicted_index]
    
    # **CORRECCIÓN CLAVE:** Extraemos el valor escalar usando .item()
    # Esto resuelve el error 'can only convert an array...'
    confidence_scalar = float((predictions[predicted_index] * 100).item()) 
    
    return prediction_class, confidence_scalar