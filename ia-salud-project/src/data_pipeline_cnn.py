import pydicom
import numpy as np
import os
from PIL import Image
import io # Necesario para leer bytes de Streamlit

# --- Configuración global ---
IMAGE_SIZE = (256, 256)

def preprocess_image_for_cnn(file_path_or_bytes, target_size=IMAGE_SIZE, from_bytes=False):
    """
    Lee y preprocesa una imagen (DICOM, PNG, JPG) para que sea compatible con la CNN.
    """
    
    try:
        img = None
        
        if from_bytes:
            # Lógica para Streamlit (bytes)
            img_source = io.BytesIO(file_path_or_bytes)
            img_pil = Image.open(img_source).convert('RGB')
            img = np.array(img_pil).astype(float)
        
        elif file_path_or_bytes.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Lógica para rutas de archivos comunes
            img_pil = Image.open(file_path_or_bytes).convert('RGB')
            img = np.array(img_pil).astype(float)
        
        elif file_path_or_bytes.lower().endswith(('.dcm')):
            # Lógica para DICOM
            ds = pydicom.dcmread(file_path_or_bytes)
            img = ds.pixel_array.astype(float)
            if img.ndim == 2:
                img = np.stack((img,)*3, axis=-1) # Replicar para 3 canales
        else:
            print("ERROR: Formato de archivo no soportado.")
            return None
        
        # Normalización y Redimensionamiento
        if img is not None:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img_pil = Image.fromarray((img * 255).astype(np.uint8)).resize(target_size)
            img = np.array(img_pil).astype(np.float32) / 255.0
            
            # Asegurar la forma final (256, 256, 3)
            return img

    except Exception as e:
        print(f"Error procesando el archivo: {e}")
        return None

if __name__ == '__main__':
    print("Módulo de preprocesamiento cargado. Necesita una ruta para probar.")