"""
================================================================================
MÓDULO DE MODELADO CON ANN (Artificial Neural Network)
Proyecto: Predicción de Deserción Académica
Responsable: Jorge
================================================================================

Este módulo contiene todas las funciones para:
1. Construir la arquitectura de la red neuronal
2. Entrenar el modelo con datos de entrenamiento
3. Guardar y cargar modelos entrenados
4. Realizar predicciones

Uso:
    from src.model_ann import build_model, train_model, load_trained_model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, Any
import os
import json
from datetime import datetime

# TensorFlow y Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, optimizers
from keras.utils import plot_model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Configurar seed para reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# =============================================================================
# 1. CONSTRUCCIÓN DEL MODELO
# =============================================================================

def build_model(input_shape: int,
                hidden_layers: list = [64, 32, 16],
                activation: str = 'relu',
                dropout_rate: float = 0.3,
                output_activation: str = 'sigmoid',
                learning_rate: float = 0.001,
                optimizer_type: str = 'adam') -> keras.Model:
    """
    Construye la arquitectura de la red neuronal para clasificación binaria.
    
    Parameters:
    -----------
    input_shape : int
        Número de features de entrada (36 en este caso)
    hidden_layers : list
        Lista con el número de neuronas por capa oculta
    activation : str
        Función de activación para capas ocultas
    dropout_rate : float
        Tasa de dropout para regularización
    output_activation : str
        Función de activación de la capa de salida ('sigmoid' para binario)
    learning_rate : float
        Tasa de aprendizaje del optimizador
    optimizer_type : str
        Tipo de optimizador ('adam', 'sgd', 'rmsprop')
        
    Returns:
    --------
    keras.Model
        Modelo compilado listo para entrenar
    """
    print("\n" + "="*60)
    print("CONSTRUCCIÓN DE LA RED NEURONAL")
    print("="*60)
    
    # Crear modelo secuencial
    model = models.Sequential(name='Dropout_Prediction_ANN')
    
    # Capa de entrada
    model.add(layers.Input(shape=(input_shape,), name='input_layer'))
    
    # Capas ocultas con Dropout
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(
            units=units,
            activation=activation,
            kernel_initializer='he_normal',  # Mejor para ReLU
            name=f'hidden_layer_{i+1}'
        ))
        # Batch Normalization para estabilizar entrenamiento
        model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
        # Dropout para prevenir overfitting
        model.add(layers.Dropout(rate=dropout_rate, name=f'dropout_{i+1}'))
    
    # Capa de salida
    model.add(layers.Dense(
        units=1,
        activation=output_activation,
        name='output_layer'
    ))
    
    # Seleccionar optimizador
    if optimizer_type == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizers.Adam(learning_rate=learning_rate)
    
    # Compilar modelo
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',  # Para clasificación binaria
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # Mostrar resumen
    print("\n📊 Arquitectura del modelo:")
    model.summary()
    
    # Calcular parámetros
    total_params = model.count_params()
    print(f"\n🔢 Total de parámetros: {total_params:,}")
    
    return model


def create_callbacks(model_save_path: str = 'models/best_model.keras',
                    patience: int = 15,
                    min_delta: float = 0.001,
                    monitor: str = 'val_loss') -> list:
    """
    Crea callbacks para el entrenamiento.
    
    Parameters:
    -----------
    model_save_path : str
        Ruta donde guardar el mejor modelo
    patience : int
        Épocas sin mejora antes de detener (early stopping)
    min_delta : float
        Cambio mínimo para considerar una mejora
    monitor : str
        Métrica a monitorear ('val_loss', 'val_accuracy', etc.)
        
    Returns:
    --------
    list
        Lista de callbacks
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callback_list = [
        # Guardar mejor modelo
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor=monitor,
            save_best_only=True,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        ),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reducir learning rate cuando se estanca
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard para visualización
        callbacks.TensorBoard(
            log_dir=f'logs/fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    return callback_list


# =============================================================================
# 2. ENTRENAMIENTO DEL MODELO
# =============================================================================

def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Calcula pesos para balancear clases desbalanceadas.
    
    Parameters:
    -----------
    y_train : np.ndarray
        Labels de entrenamiento
        
    Returns:
    --------
    dict
        Diccionario con pesos por clase
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights = dict(zip(classes, weights))
    
    print("\n⚖️ Pesos de clase calculados:")
    for clase, peso in class_weights.items():
        clase_name = "Dropout" if clase == 1 else "No Dropout"
        print(f"   - {clase_name} ({clase}): {peso:.3f}")
    
    return class_weights


def train_model(model: keras.Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                epochs: int = 100,
                batch_size: int = 32,
                use_class_weights: bool = True,
                model_save_path: str = 'models/best_model.keras',
                verbose: int = 1) -> Dict[str, Any]:
    """
    Entrena la red neuronal.
    
    Parameters:
    -----------
    model : keras.Model
        Modelo a entrenar
    X_train : np.ndarray
        Features de entrenamiento
    y_train : np.ndarray
        Labels de entrenamiento
    X_val : np.ndarray
        Features de validación
    y_val : np.ndarray
        Labels de validación
    epochs : int
        Número máximo de épocas
    batch_size : int
        Tamaño del batch
    use_class_weights : bool
        Si es True, usa pesos para balancear clases
    model_save_path : str
        Ruta para guardar el mejor modelo
    verbose : int
        Nivel de verbosidad (0, 1, 2)
        
    Returns:
    --------
    dict
        Diccionario con el historial de entrenamiento y métricas
    """
    print("\n" + "="*60)
    print("ENTRENAMIENTO DE LA RED NEURONAL")
    print("="*60)
    
    print(f"\n📊 Configuración:")
    print(f"   - Épocas máximas: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Training samples: {len(X_train):,}")
    print(f"   - Validation samples: {len(X_val):,}")
    
    # Calcular class weights si está habilitado
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(y_train)
    
    # Crear callbacks
    callback_list = create_callbacks(
        model_save_path=model_save_path,
        patience=15
    )
    
    # Entrenar modelo
    print("\n🚀 Iniciando entrenamiento...")
    start_time = datetime.now()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        verbose=verbose
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"\n✅ Entrenamiento completado en {training_time:.2f} segundos")
    
    # Obtener mejor época
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])
    best_val_acc = history.history['val_accuracy'][best_epoch - 1]
    
    print(f"\n🏆 Mejor modelo:")
    print(f"   - Época: {best_epoch}/{len(history.history['loss'])}")
    print(f"   - Val Loss: {best_val_loss:.4f}")
    print(f"   - Val Accuracy: {best_val_acc:.4f}")
    
    # Guardar historial
    history_dict = {
        'history': history.history,
        'best_epoch': best_epoch,
        'training_time_seconds': training_time,
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'use_class_weights': use_class_weights
        }
    }
    
    # Guardar historial como JSON
    history_path = model_save_path.replace('.keras', '_history.json')
    with open(history_path, 'w') as f:
        # Convertir arrays numpy a listas para JSON
        history_json = {}
        for k, v in history_dict.items():
            if k == 'config':
                history_json[k] = history_dict[k]
            elif k == 'best_epoch':
                history_json[k] = int(v)  # Convertir numpy.int64 a int
            elif k == 'training_time_seconds':
                history_json[k] = float(v)  # Asegurar que es float
            elif k == 'history':
                # Convertir todos los valores del historial
                history_json[k] = {
                    metric: [float(x) for x in values]
                    for metric, values in v.items()
                }
        json.dump(history_json, f, indent=2)
    
    print(f"💾 Historial guardado: {history_path}")
    
    return history_dict


# =============================================================================
# 3. VISUALIZACIÓN DEL ENTRENAMIENTO
# =============================================================================

def plot_training_history(history_dict: Dict[str, Any],
                            save_path: str = None) -> plt.Figure:
    """
    Visualiza el historial de entrenamiento.
    
    Parameters:
    -----------
    history_dict : dict
        Diccionario con el historial de entrenamiento
    save_path : str, optional
        Ruta donde guardar la figura
        
    Returns:
    --------
    plt.Figure
        Figura con los gráficos
    """
    history = history_dict['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Historial de Entrenamiento', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss por Época')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy por Época')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Train Precision', linewidth=2)
        axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision por Época')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall por Época')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {save_path}")
    
    return fig


# =============================================================================
# 4. GUARDAR Y CARGAR MODELOS
# =============================================================================

def save_model_complete(model: keras.Model,
                        model_path: str = 'models/ann_dropout.keras',
                        save_architecture: bool = True) -> None:
    """
    Guarda el modelo completo con metadata adicional.
    
    Parameters:
    -----------
    model : keras.Model
        Modelo a guardar
    model_path : str
        Ruta donde guardar el modelo
    save_architecture : bool
        Si es True, guarda también la arquitectura como imagen
    """
    # Guardar modelo
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"✅ Modelo guardado: {model_path}")
    
    # Guardar arquitectura como imagen
    if save_architecture:
        try:
            arch_path = model_path.replace('.keras', '_architecture.png')
            plot_model(
                model,
                to_file=arch_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                dpi=150
            )
            print(f"✅ Arquitectura guardada: {arch_path}")
        except Exception as e:
            print(f"⚠️ No se pudo guardar la arquitectura: {e}")
    
    # Guardar metadata
    metadata = {
        'input_shape': model.input_shape[1:],
        'output_shape': model.output_shape[1:],
        'total_params': int(model.count_params()),
        'layers': [
            {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'config': str(layer.get_config())
            }
            for layer in model.layers
        ],
        'saved_at': datetime.now().isoformat()
    }
    
    metadata_path = model_path.replace('.keras', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata guardada: {metadata_path}")


def load_trained_model(model_path: str = 'models/best_model.keras') -> keras.Model:
    """
    Carga un modelo entrenado.
    
    Parameters:
    -----------
    model_path : str
        Ruta al archivo del modelo
        
    Returns:
    --------
    keras.Model
        Modelo cargado
        
    Raises:
    -------
    FileNotFoundError
        Si el modelo no existe
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    print(f"📂 Cargando modelo desde: {model_path}")
    model = keras.models.load_model(model_path)
    
    print(f"✅ Modelo cargado exitosamente")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
    print(f"   - Total params: {model.count_params():,}")
    
    return model


# =============================================================================
# 5. PREDICCIÓN
# =============================================================================

def predict_dropout(model: keras.Model,
                    X: np.ndarray,
                    threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza predicciones de deserción.
    
    Parameters:
    -----------
    model : keras.Model
        Modelo entrenado
    X : np.ndarray
        Features de entrada
    threshold : float
        Umbral para clasificación (default: 0.5)
        
    Returns:
    --------
    tuple
        (probabilidades, clases_predichas)
    """
    # Obtener probabilidades
    probabilities = model.predict(X, verbose=0)
    
    # Convertir a clases usando el umbral
    predictions = (probabilities >= threshold).astype(int).flatten()
    
    return probabilities.flatten(), predictions


def predict_single_student(model: keras.Model,
                            student_features: np.ndarray,
                            feature_names: list = None) -> Dict[str, Any]:
    """
    Predice el riesgo de deserción para un solo estudiante.
    
    Parameters:
    -----------
    model : keras.Model
        Modelo entrenado
    student_features : np.ndarray
        Features del estudiante (debe estar normalizado)
    feature_names : list, optional
        Nombres de las features
        
    Returns:
    --------
    dict
        Resultado de la predicción con interpretación
    """
    # Asegurar que sea 2D
    if student_features.ndim == 1:
        student_features = student_features.reshape(1, -1)
    
    # Predecir
    prob, pred = predict_dropout(model, student_features)
    
    # Interpretar resultado
    risk_level = "ALTO" if prob[0] >= 0.7 else "MEDIO" if prob[0] >= 0.4 else "BAJO"
    will_dropout = "Sí" if pred[0] == 1 else "No"
    
    resultado = {
        'probabilidad_desercion': float(prob[0]),
        'prediccion': int(pred[0]),
        'prediccion_texto': will_dropout,
        'nivel_riesgo': risk_level,
        'interpretacion': f"El estudiante tiene un riesgo {risk_level} de deserción "
                            f"(probabilidad: {prob[0]*100:.1f}%)"
    }
    
    return resultado


# =============================================================================
# 6. PIPELINE COMPLETO DE ENTRENAMIENTO
# =============================================================================

def train_pipeline(data_dir: str = 'data/processed',
                    model_dir: str = 'models',
                    epochs: int = 100,
                    batch_size: int = 32,
                    hidden_layers: list = [64, 32, 16],
                    dropout_rate: float = 0.3,
                    learning_rate: float = 0.001) -> Dict[str, Any]:
    """
    Pipeline completo de entrenamiento.
    
    Parameters:
    -----------
    data_dir : str
        Directorio con los datos procesados
    model_dir : str
        Directorio donde guardar el modelo
    epochs : int
        Número de épocas
    batch_size : int
        Tamaño del batch
    hidden_layers : list
        Configuración de capas ocultas
    dropout_rate : float
        Tasa de dropout
    learning_rate : float
        Learning rate
        
    Returns:
    --------
    dict
        Resultados del pipeline
    """
    print("="*60)
    print("PIPELINE COMPLETO DE ENTRENAMIENTO")
    print("="*60)
    
    # 1. Cargar datos
    print("\n📂 Cargando datos...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_val: {X_val.shape}")
    
    # 2. Construir modelo
    model = build_model(
        input_shape=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # 3. Entrenar
    model_path = os.path.join(model_dir, 'best_model.keras')
    history_dict = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_save_path=model_path
    )
    
    # 4. Cargar mejor modelo
    best_model = load_trained_model(model_path)
    
    # 5. Visualizar entrenamiento
    plot_path = os.path.join(model_dir, 'training_history.png')
    plot_training_history(history_dict, save_path=plot_path)
    
    # 6. Guardar modelo final con metadata
    final_model_path = os.path.join(model_dir, 'ann_dropout.keras')
    save_model_complete(best_model, final_model_path, save_architecture=True)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETADO")
    print("="*60)
    
    return {
        'model': best_model,
        'history': history_dict,
        'model_path': final_model_path
    }


# =============================================================================
# EJECUCIÓN COMO SCRIPT
# =============================================================================

if __name__ == "__main__":
    # Ejecutar pipeline completo
    print("🚀 Iniciando entrenamiento de la ANN...")
    
    results = train_pipeline(
        data_dir='data/processed',
        model_dir='models',
        epochs=100,
        batch_size=32,
        hidden_layers=[64, 32, 16],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    print("\n📁 Archivos generados:")
    print("   - models/best_model.keras (mejor modelo durante entrenamiento)")
    print("   - models/ann_dropout.keras (modelo final)")
    print("   - models/ann_dropout_architecture.png")
    print("   - models/ann_dropout_metadata.json")
    print("   - models/best_model_history.json")
    print("   - models/training_history.png")
    
    print("\n💡 Para usar el modelo:")
    print("   from src.model_ann import load_trained_model, predict_single_student")
    print("   model = load_trained_model('models/ann_dropout.keras')")
    print("   resultado = predict_single_student(model, student_data)")