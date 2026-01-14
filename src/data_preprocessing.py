"""
================================================================================
MÓDULO DE PREPROCESAMIENTO DE DATOS
Proyecto: Predicción de Deserción Académica con ANN
Responsable: David
================================================================================

Este módulo contiene todas las funciones necesarias para:
1. Cargar el dataset
2. Realizar análisis exploratorio (EDA)
3. Preprocesar los datos (limpieza, codificación, normalización)
4. Dividir los datos en conjuntos de entrenamiento y prueba

Uso:
    from src.data_preprocessing import load_data, preprocess_data, split_data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import warnings
import os

warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# 1. FUNCIONES DE CARGA DE DATOS
# =============================================================================

def load_data(filepath: str = 'data/raw/students_dropout.csv') -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV con los datos
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con los datos cargados
        
    Raises:
    -------
    FileNotFoundError
        Si el archivo no existe
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    
    # Intentar detectar el separador automáticamente
    # El dataset de UCI usa punto y coma (;) como separador
    try:
        # Primero intentar con punto y coma (formato UCI)
        df = pd.read_csv(filepath, sep=';')
        if df.shape[1] == 1:
            # Si solo hay una columna, probablemente el separador es coma
            df = pd.read_csv(filepath, sep=',')
    except:
        df = pd.read_csv(filepath)
    
    # Limpiar nombres de columnas (quitar espacios extra y caracteres especiales)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
    df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
    
    # Mapeo de nombres de columnas del dataset UCI al formato estándar
    column_mapping = {
        'Marital_status': 'Marital_status',
        'Application_mode': 'Application_mode',
        'Application_order': 'Application_order',
        'Course': 'Course',
        'Daytimeevening_attendance': 'Daytime_evening_attendance',
        'Daytime_evening_attendance': 'Daytime_evening_attendance',
        'Previous_qualification': 'Previous_qualification',
        'Previous_qualification_grade': 'Previous_qualification_grade',
        'Nacionality': 'Nacionality',
        'Mothers_qualification': 'Mothers_qualification',
        'Fathers_qualification': 'Fathers_qualification',
        'Mothers_occupation': 'Mothers_occupation',
        'Fathers_occupation': 'Fathers_occupation',
        'Admission_grade': 'Admission_grade',
        'Displaced': 'Displaced',
        'Educational_special_needs': 'Educational_special_needs',
        'Debtor': 'Debtor',
        'Tuition_fees_up_to_date': 'Tuition_fees_up_to_date',
        'Gender': 'Gender',
        'Scholarship_holder': 'Scholarship_holder',
        'Age_at_enrollment': 'Age_at_enrollment',
        'International': 'International',
        'Curricular_units_1st_sem_credited': 'Curricular_units_1st_sem_credited',
        'Curricular_units_1st_sem_enrolled': 'Curricular_units_1st_sem_enrolled',
        'Curricular_units_1st_sem_evaluations': 'Curricular_units_1st_sem_evaluations',
        'Curricular_units_1st_sem_approved': 'Curricular_units_1st_sem_approved',
        'Curricular_units_1st_sem_grade': 'Curricular_units_1st_sem_grade',
        'Curricular_units_1st_sem_without_evaluations': 'Curricular_units_1st_sem_without_evaluations',
        'Curricular_units_2nd_sem_credited': 'Curricular_units_2nd_sem_credited',
        'Curricular_units_2nd_sem_enrolled': 'Curricular_units_2nd_sem_enrolled',
        'Curricular_units_2nd_sem_evaluations': 'Curricular_units_2nd_sem_evaluations',
        'Curricular_units_2nd_sem_approved': 'Curricular_units_2nd_sem_approved',
        'Curricular_units_2nd_sem_grade': 'Curricular_units_2nd_sem_grade',
        'Curricular_units_2nd_sem_without_evaluations': 'Curricular_units_2nd_sem_without_evaluations',
        'Unemployment_rate': 'Unemployment_rate',
        'Inflation_rate': 'Inflation_rate',
        'GDP': 'GDP',
        'Target': 'Target'
    }
    
    # Renombrar columnas que coincidan
    df.columns = [column_mapping.get(col, col) for col in df.columns]
    
    print(f"✓ Dataset cargado exitosamente")
    print(f"  - Filas: {df.shape[0]:,}")
    print(f"  - Columnas: {df.shape[1]}")
    
    return df


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Obtiene información general del dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
        
    Returns:
    --------
    dict
        Diccionario con información del dataset
    """
    info = {
        'n_filas': df.shape[0],
        'n_columnas': df.shape[1],
        'columnas': list(df.columns),
        'tipos_datos': df.dtypes.value_counts().to_dict(),
        'valores_nulos': df.isnull().sum().sum(),
        'memoria_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    return info


# =============================================================================
# 2. FUNCIONES DE ANÁLISIS EXPLORATORIO (EDA)
# =============================================================================

def eda_resumen(df: pd.DataFrame) -> None:
    """
    Muestra un resumen estadístico completo del dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
    """
    print("\n" + "="*60)
    print("RESUMEN DEL DATASET")
    print("="*60)
    
    # Información general
    print(f"\n📊 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"💾 Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Valores nulos
    nulos = df.isnull().sum()
    if nulos.sum() == 0:
        print("✓ Sin valores nulos")
    else:
        print(f"⚠ Valores nulos encontrados:")
        print(nulos[nulos > 0])
    
    # Tipos de datos
    print(f"\n📋 Tipos de datos:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"   - {dtype}: {count} columnas")
    
    # Distribución del target
    if 'Target' in df.columns:
        print(f"\n🎯 Distribución del Target:")
        for target, count in df['Target'].value_counts().items():
            pct = count / len(df) * 100
            print(f"   - {target}: {count:,} ({pct:.1f}%)")


def eda_visualizaciones(df: pd.DataFrame, save_path: str = None) -> Dict[str, plt.Figure]:
    """
    Genera visualizaciones del análisis exploratorio.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
    save_path : str, optional
        Ruta donde guardar las figuras
        
    Returns:
    --------
    dict
        Diccionario con las figuras generadas
    """
    figuras = {}
    
    # 1. Distribución del Target
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # Rojo, Amarillo, Verde
    target_counts = df['Target'].value_counts()
    bars = ax1.bar(target_counts.index, target_counts.values, color=colors, edgecolor='black')
    ax1.set_xlabel('Estado del Estudiante', fontsize=12)
    ax1.set_ylabel('Cantidad', fontsize=12)
    ax1.set_title('Distribución de Deserción Académica', fontsize=14, fontweight='bold')
    
    # Añadir etiquetas en las barras
    for bar, count in zip(bars, target_counts.values):
        pct = count / len(df) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    figuras['target_distribution'] = fig1
    
    # 2. Matriz de correlación (variables numéricas)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        fig2, ax2 = plt.subplots(figsize=(16, 14))
        corr_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', 
                   center=0, ax=ax2, cbar_kws={'label': 'Correlación'})
        ax2.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
        plt.tight_layout()
        figuras['correlation_matrix'] = fig2
    
    # 3. Distribución de edad por Target
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for target in df['Target'].unique():
        subset = df[df['Target'] == target]['Age_at_enrollment']
        ax3.hist(subset, bins=30, alpha=0.5, label=target, edgecolor='black')
    ax3.set_xlabel('Edad al Inscribirse', fontsize=12)
    ax3.set_ylabel('Frecuencia', fontsize=12)
    ax3.set_title('Distribución de Edad por Estado del Estudiante', fontsize=14, fontweight='bold')
    ax3.legend()
    plt.tight_layout()
    figuras['age_distribution'] = fig3
    
    # 4. Rendimiento académico por semestre
    fig4, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Primer semestre
    df.boxplot(column='Curricular_units_1st_sem_approved', by='Target', ax=axes[0])
    axes[0].set_title('Materias Aprobadas - 1er Semestre', fontsize=12)
    axes[0].set_xlabel('Estado')
    axes[0].set_ylabel('Materias Aprobadas')
    
    # Segundo semestre
    df.boxplot(column='Curricular_units_2nd_sem_approved', by='Target', ax=axes[1])
    axes[1].set_title('Materias Aprobadas - 2do Semestre', fontsize=12)
    axes[1].set_xlabel('Estado')
    axes[1].set_ylabel('Materias Aprobadas')
    
    fig4.suptitle('Rendimiento Académico por Estado del Estudiante', fontsize=14, fontweight='bold')
    plt.tight_layout()
    figuras['academic_performance'] = fig4
    
    # 5. Variables socioeconómicas
    fig5, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Becarios
    pd.crosstab(df['Target'], df['Scholarship_holder']).plot(kind='bar', ax=axes[0,0], 
               color=['#3498db', '#e74c3c'])
    axes[0,0].set_title('Becarios por Estado')
    axes[0,0].set_xlabel('Estado del Estudiante')
    axes[0,0].legend(['No Becario', 'Becario'])
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Deudores
    pd.crosstab(df['Target'], df['Debtor']).plot(kind='bar', ax=axes[0,1],
               color=['#27ae60', '#e74c3c'])
    axes[0,1].set_title('Deudores por Estado')
    axes[0,1].set_xlabel('Estado del Estudiante')
    axes[0,1].legend(['No Deudor', 'Deudor'])
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Matrícula al día
    pd.crosstab(df['Target'], df['Tuition_fees_up_to_date']).plot(kind='bar', ax=axes[1,0],
               color=['#e74c3c', '#27ae60'])
    axes[1,0].set_title('Matrícula al Día por Estado')
    axes[1,0].set_xlabel('Estado del Estudiante')
    axes[1,0].legend(['Atrasado', 'Al Día'])
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Género
    pd.crosstab(df['Target'], df['Gender']).plot(kind='bar', ax=axes[1,1],
               color=['#9b59b6', '#3498db'])
    axes[1,1].set_title('Género por Estado')
    axes[1,1].set_xlabel('Estado del Estudiante')
    axes[1,1].legend(['Femenino', 'Masculino'])
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    figuras['socioeconomic'] = fig5
    
    # Guardar figuras si se especifica ruta
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        for name, fig in figuras.items():
            fig.savefig(os.path.join(save_path, f'{name}.png'), dpi=150, bbox_inches='tight')
            print(f"✓ Guardado: {name}.png")
    
    return figuras


def analizar_outliers(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Analiza outliers en las columnas numéricas usando IQR.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
    columns : list, optional
        Lista de columnas a analizar. Si es None, usa todas las numéricas.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con información de outliers por columna
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = []
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct_outliers = n_outliers / len(df) * 100
        
        outlier_info.append({
            'columna': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'limite_inferior': lower,
            'limite_superior': upper,
            'n_outliers': n_outliers,
            'pct_outliers': round(pct_outliers, 2)
        })
    
    return pd.DataFrame(outlier_info)


# =============================================================================
# 3. FUNCIONES DE PREPROCESAMIENTO
# =============================================================================

def preprocess_data(df: pd.DataFrame, 
                   target_column: str = 'Target',
                   binary_classification: bool = True,
                   scale_features: bool = True) -> Tuple[pd.DataFrame, pd.Series, StandardScaler, LabelEncoder]:
    """
    Preprocesa los datos para el modelo de red neuronal.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos originales
    target_column : str
        Nombre de la columna objetivo
    binary_classification : bool
        Si es True, convierte a clasificación binaria (Dropout vs No Dropout)
    scale_features : bool
        Si es True, normaliza las features numéricas
        
    Returns:
    --------
    tuple
        (X, y, scaler, label_encoder)
        - X: Features preprocesadas
        - y: Target codificado
        - scaler: Objeto StandardScaler ajustado (o None)
        - label_encoder: Objeto LabelEncoder ajustado
    """
    print("\n" + "="*60)
    print("PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    df_processed = df.copy()
    
    # 1. Separar features y target
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column].copy()
    
    # 2. Convertir a clasificación binaria si se requiere
    if binary_classification:
        # Dropout = 1, No Dropout (Graduate/Enrolled) = 0
        y = y.apply(lambda x: 1 if x == 'Dropout' else 0)
        print("✓ Convertido a clasificación binaria:")
        print(f"   - Dropout (1): {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
        print(f"   - No Dropout (0): {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
        label_encoder = None
    else:
        # Codificar con LabelEncoder para multiclase
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=target_column)
        print("✓ Target codificado (multiclase):")
        for i, clase in enumerate(label_encoder.classes_):
            count = (y == i).sum()
            print(f"   - {clase} ({i}): {count:,}")
    
    # 3. Manejar valores nulos (si los hay)
    null_counts = X.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\n⚠ Manejando {null_counts.sum()} valores nulos...")
        # Imputar numéricos con la mediana
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        # Imputar categóricos con la moda
        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
        print("✓ Valores nulos imputados")
    else:
        print("✓ Sin valores nulos")
    
    # 4. Normalizar features numéricas
    if scale_features:
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        print(f"✓ {len(numeric_cols)} columnas numéricas normalizadas (StandardScaler)")
    else:
        scaler = None
        print("✓ Sin normalización aplicada")
    
    print(f"\n📊 Dimensiones finales:")
    print(f"   - X: {X.shape}")
    print(f"   - y: {y.shape}")
    
    return X, y, scaler, label_encoder


def split_data(X: pd.DataFrame, 
               y: pd.Series,
               test_size: float = 0.2,
               val_size: float = 0.1,
               random_state: int = 42,
               stratify: bool = True) -> Dict[str, np.ndarray]:
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Proporción para conjunto de prueba
    val_size : float
        Proporción para conjunto de validación (del total)
    random_state : int
        Semilla para reproducibilidad
    stratify : bool
        Si es True, mantiene la proporción de clases
        
    Returns:
    --------
    dict
        Diccionario con X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\n" + "="*60)
    print("DIVISIÓN DE DATOS")
    print("="*60)
    
    strat = y if stratify else None
    
    # Primera división: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=strat
    )
    
    # Segunda división: train vs val
    # Ajustar proporción de validación respecto al conjunto temporal
    val_ratio = val_size / (1 - test_size)
    strat_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=strat_temp
    )
    
    # Convertir a numpy arrays
    data = {
        'X_train': X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
        'X_val': X_val.values if isinstance(X_val, pd.DataFrame) else X_val,
        'X_test': X_test.values if isinstance(X_test, pd.DataFrame) else X_test,
        'y_train': y_train.values if isinstance(y_train, pd.Series) else y_train,
        'y_val': y_val.values if isinstance(y_val, pd.Series) else y_val,
        'y_test': y_test.values if isinstance(y_test, pd.Series) else y_test,
    }
    
    print(f"✓ División completada:")
    print(f"   - Entrenamiento: {len(data['X_train']):,} muestras ({len(data['X_train'])/len(X)*100:.1f}%)")
    print(f"   - Validación: {len(data['X_val']):,} muestras ({len(data['X_val'])/len(X)*100:.1f}%)")
    print(f"   - Prueba: {len(data['X_test']):,} muestras ({len(data['X_test'])/len(X)*100:.1f}%)")
    
    if stratify:
        print(f"\n📊 Distribución de clases (estratificada):")
        for name, y_subset in [('Train', data['y_train']), ('Val', data['y_val']), ('Test', data['y_test'])]:
            if len(np.unique(y_subset)) == 2:  # Binario
                pct_pos = np.mean(y_subset) * 100
                print(f"   - {name}: {pct_pos:.1f}% positivos (Dropout)")
            else:  # Multiclase
                print(f"   - {name}: {dict(zip(*np.unique(y_subset, return_counts=True)))}")
    
    return data


def save_processed_data(data: Dict[str, np.ndarray], 
                       output_dir: str = 'data/processed',
                       prefix: str = '') -> None:
    """
    Guarda los datos procesados en archivos numpy.
    
    Parameters:
    -----------
    data : dict
        Diccionario con los arrays de datos
    output_dir : str
        Directorio de salida
    prefix : str
        Prefijo para los nombres de archivo
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, array in data.items():
        filename = f"{prefix}{name}.npy" if prefix else f"{name}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, array)
    
    print(f"\n✓ Datos guardados en {output_dir}/")


def load_processed_data(input_dir: str = 'data/processed',
                       prefix: str = '') -> Dict[str, np.ndarray]:
    """
    Carga los datos procesados desde archivos numpy.
    
    Parameters:
    -----------
    input_dir : str
        Directorio de entrada
    prefix : str
        Prefijo de los nombres de archivo
        
    Returns:
    --------
    dict
        Diccionario con los arrays de datos
    """
    data = {}
    expected_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
    
    for key in expected_keys:
        filename = f"{prefix}{key}.npy" if prefix else f"{key}.npy"
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            data[key] = np.load(filepath)
    
    print(f"✓ Datos cargados desde {input_dir}/")
    return data


# =============================================================================
# 4. FUNCIÓN PRINCIPAL (PIPELINE COMPLETO)
# =============================================================================

def run_preprocessing_pipeline(raw_data_path: str = 'data/raw/students_dropout.csv',
                               output_dir: str = 'data/processed',
                               binary: bool = True,
                               save_plots: bool = True) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de preprocesamiento.
    
    Parameters:
    -----------
    raw_data_path : str
        Ruta al archivo CSV con datos crudos
    output_dir : str
        Directorio para guardar datos procesados
    binary : bool
        Si es True, usa clasificación binaria
    save_plots : bool
        Si es True, guarda las visualizaciones
        
    Returns:
    --------
    dict
        Diccionario con datos procesados y objetos del preprocesamiento
    """
    print("="*60)
    print("PIPELINE DE PREPROCESAMIENTO - DESERCIÓN ACADÉMICA")
    print("="*60)
    
    # 1. Cargar datos
    df = load_data(raw_data_path)
    
    # 2. EDA
    eda_resumen(df)
    
    plot_dir = os.path.join(output_dir, 'plots') if save_plots else None
    figuras = eda_visualizaciones(df, save_path=plot_dir)
    
    # 3. Preprocesar
    X, y, scaler, label_encoder = preprocess_data(
        df, 
        binary_classification=binary,
        scale_features=True
    )
    
    # 4. Dividir datos
    data_splits = split_data(X, y)
    
    # 5. Guardar
    save_processed_data(data_splits, output_dir)
    
    # Guardar feature names para uso posterior
    feature_names = X.columns.tolist()
    np.save(os.path.join(output_dir, 'feature_names.npy'), feature_names)
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return {
        'data': data_splits,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'figuras': figuras
    }


# =============================================================================
# EJECUCIÓN COMO SCRIPT
# =============================================================================

if __name__ == "__main__":
    # Ejecutar pipeline completo
    result = run_preprocessing_pipeline(
        raw_data_path='data/raw/students_dropout.csv',
        output_dir='data/processed',
        binary=True,
        save_plots=True
    )
    
    print("\n📁 Archivos generados:")
    print("   - data/processed/X_train.npy")
    print("   - data/processed/X_val.npy")
    print("   - data/processed/X_test.npy")
    print("   - data/processed/y_train.npy")
    print("   - data/processed/y_val.npy")
    print("   - data/processed/y_test.npy")
    print("   - data/processed/feature_names.npy")
    print("   - data/processed/plots/*.png")