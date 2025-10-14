import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import ResNet50 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import sys

# --- Configuración de Rutas y Hiperparámetros ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Hiperparámetros
NUM_CLASSES = 2  
BATCH_SIZE = 2   
EPOCHS = 10 
IMAGE_SIZE = (256, 256)
INPUT_SHAPE = IMAGE_SIZE + (3,) 

# ----------------------------------------------------------------------
# 1. Generador de Datos
# ----------------------------------------------------------------------

def load_data_generator(data_dir):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        directory=os.path.join(data_dir, 'imagenes_dicom'),
        target_size=IMAGE_SIZE,
        color_mode='rgb', 
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        directory=os.path.join(data_dir, 'imagenes_dicom'),
        target_size=IMAGE_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

# ----------------------------------------------------------------------
# 2. Definición del Modelo
# ----------------------------------------------------------------------

def build_transfer_model(input_shape, num_classes):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    for layer in base_model.layers:
        layer.trainable = False
        
    model = Sequential([
        base_model,
        Flatten(), 
        Dense(256, activation='relu'),
        Dropout(0.5), 
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ----------------------------------------------------------------------
# 3. Entrenamiento y Guardado
# ----------------------------------------------------------------------

def train_cnn():
    print("--- 🔬 Iniciando Entrenamiento de la CNN (Transfer Learning) ---")
    
    train_generator, validation_generator = load_data_generator(DATA_DIR)
    
    if train_generator.num_classes != NUM_CLASSES:
         print(f"ERROR: Se esperaba {NUM_CLASSES} clases, pero el generador encontró {train_generator.num_classes}.")
         return

    model = build_transfer_model(INPUT_SHAPE, NUM_CLASSES)
    model.summary()
    
    print(f"\nEntrenando con {train_generator.samples} muestras en {EPOCHS} épocas...")
    
    # Cálculo robusto de pasos
    steps_per_epoch = int(np.ceil(train_generator.samples / BATCH_SIZE))
    validation_steps = int(np.ceil(validation_generator.samples / BATCH_SIZE))
    
    validation_data = validation_generator
    if validation_steps == 0 and validation_generator.samples > 0:
        validation_steps = 1
    elif validation_generator.samples == 0:
        print("⚠️ Advertencia: Cero imágenes para validación. El entrenamiento continuará sin validación.")
        validation_steps = None
        validation_data = None

    # Entrenar el modelo
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_data,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # 💾 Guardar el modelo de diagnóstico
    model_save_path = os.path.join(MODELS_DIR, 'cnn_diagnostico.h5')
    model.save(model_save_path)
    print(f"\n✅ Modelo CNN guardado exitosamente en: {model_save_path}")

    # --- CORRECCIÓN FINAL para el Extractor de Características ---
    # Guardamos la capa base (ResNet50) directamente para evitar el error 'inputs not connected to outputs'.
    
    # 1. Obtener la capa ResNet50 (es la primera capa del Sequential model)
    feature_extractor = model.layers[0] 

    extractor_save_path = os.path.join(MODELS_DIR, 'cnn_feature_extractor.h5')
    
    # 2. Guardar la capa base directamente.
    feature_extractor.save(extractor_save_path) 
    print(f"✅ Extractor de Features guardado para futura búsqueda semántica: {extractor_save_path}")


if __name__ == '__main__':
    train_cnn()