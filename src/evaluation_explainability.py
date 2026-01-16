"""
Módulo de evaluación y explicabilidad para el modelo de predicción de deserción académica.
Contiene funciones para calcular métricas, visualizar matriz de confusión y curva ROC.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
from sklearn.inspection import permutation_importance


def evaluate_model(model, X_test, y_test):
    """
    Calcula métricas de clasificación para el modelo.
    
    Args:
        model: Modelo Keras entrenado
        X_test: Datos de prueba (features)
        y_test: Etiquetas reales de prueba
        
    Returns:
        dict: Diccionario con todas las métricas de clasificación
            - accuracy: Exactitud del modelo
            - precision: Precisión (positivos predichos correctamente)
            - recall: Recall/Sensibilidad (positivos reales detectados)
            - f1_score: Media armónica de precision y recall
            - roc_auc: Área bajo la curva ROC
    """
    # Obtener predicciones de probabilidad
    y_pred_proba = model.predict(X_test, verbose=0)
    
    # Convertir probabilidades a clases (umbral 0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    
    # Asegurar que y_test sea un array plano
    y_true = np.array(y_test).flatten()
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    
    # Imprimir resumen
    print("=" * 50)
    print("MÉTRICAS DE EVALUACIÓN DEL MODELO")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 50)
    
    # Mostrar reporte de clasificación completo
    print("\nReporte de Clasificación Detallado:")
    print("-" * 50)
    target_names = ['No Deserta (0)', 'Deserta (1)']
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Genera una matriz de confusión visual usando seaborn.
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones del modelo (clases, no probabilidades)
        save_path: Ruta para guardar la imagen (opcional)
        
    Returns:
        numpy.ndarray: Matriz de confusión
    """
    # Asegurar arrays planos
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Crear figura
    plt.figure(figsize=(8, 6))
    
    # Crear heatmap con seaborn
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Deserta (0)', 'Deserta (1)'],
        yticklabels=['No Deserta (0)', 'Deserta (1)'],
        annot_kws={'size': 14}
    )
    
    plt.title('Matriz de Confusión\nPredicción de Deserción Académica', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Valor Predicho', fontsize=12)
    
    # Añadir texto explicativo
    plt.text(0.5, -0.15, 
             f'TN={cm[0,0]} | FP={cm[0,1]} | FN={cm[1,0]} | TP={cm[1,1]}',
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")
    
    plt.show()
    
    return cm


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Genera la curva ROC con el área bajo la curva.
    
    Args:
        y_true: Etiquetas reales
        y_pred_proba: Probabilidades predichas por el modelo
        save_path: Ruta para guardar la imagen (opcional)
        
    Returns:
        float: Área bajo la curva ROC (AUC)
    """
    # Asegurar arrays planos
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()
    
    # Calcular curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calcular AUC
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Crear figura
    plt.figure(figsize=(8, 6))
    
    # Graficar curva ROC
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'Curva ROC (AUC = {auc:.4f})')
    
    # Línea diagonal (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Clasificador Aleatorio (AUC = 0.50)')
    
    # Configurar gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title('Curva ROC - Predicción de Deserción Académica', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Añadir punto óptimo (opcional - punto más cercano a esquina superior izquierda)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
                label=f'Umbral óptimo = {optimal_threshold:.3f}', zorder=5)
    plt.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curva ROC guardada en: {save_path}")
    
    plt.show()
    
    return auc


# =============================================================================
# FUNCIONES DE EXPLICABILIDAD
# =============================================================================

def get_feature_importance(model, X_test, y_test, feature_names, n_repeats=10, random_state=42):
    """
    Calcula la importancia de características usando permutation importance.
    Implementación manual compatible con modelos Keras.
    
    Args:
        model: Modelo Keras entrenado
        X_test: Datos de prueba (features)
        y_test: Etiquetas reales de prueba
        feature_names: Lista o array con nombres de las características
        n_repeats: Número de repeticiones para el cálculo (default: 10)
        random_state: Semilla para reproducibilidad
        
    Returns:
        pandas.DataFrame: DataFrame ordenado con importancia de cada característica
            Columnas: feature, importance_mean, importance_std
            
    Raises:
        ValueError: Si las dimensiones no coinciden
        RuntimeError: Si hay error en el cálculo de permutation importance
    """
    try:
        # Validar dimensiones
        if X_test.shape[1] != len(feature_names):
            raise ValueError(
                f"Número de características ({X_test.shape[1]}) no coincide "
                f"con número de nombres ({len(feature_names)})"
            )
        
        # Asegurar que y_test sea array plano
        y_true = np.array(y_test).flatten()
        X_data = np.array(X_test).copy()
        
        print("Calculando importancia de características...")
        print(f"Esto puede tomar un momento ({n_repeats} repeticiones)...")
        
        # Establecer semilla
        np.random.seed(random_state)
        
        # Calcular score base (accuracy)
        y_pred_base = (model.predict(X_data, verbose=0) >= 0.5).astype(int).flatten()
        base_score = accuracy_score(y_true, y_pred_base)
        
        # Calcular importancia para cada característica
        n_features = X_data.shape[1]
        importances = np.zeros((n_features, n_repeats))
        
        for feat_idx in range(n_features):
            for rep in range(n_repeats):
                # Crear copia y permutar la característica
                X_permuted = X_data.copy()
                X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
                
                # Calcular score con característica permutada
                y_pred_perm = (model.predict(X_permuted, verbose=0) >= 0.5).astype(int).flatten()
                perm_score = accuracy_score(y_true, y_pred_perm)
                
                # La importancia es la caída en el score
                importances[feat_idx, rep] = base_score - perm_score
        
        # Calcular media y desviación estándar
        importance_mean = importances.mean(axis=1)
        importance_std = importances.std(axis=1)
        
        # Crear DataFrame con resultados
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': importance_mean,
            'importance_std': importance_std
        })
        
        # Ordenar por importancia descendente
        importance_df = importance_df.sort_values(
            'importance_mean', 
            ascending=False
        ).reset_index(drop=True)
        
        print("\n" + "=" * 50)
        print("IMPORTANCIA DE CARACTERÍSTICAS (Top 10)")
        print("=" * 50)
        print(importance_df.head(10).to_string(index=False))
        print("=" * 50)
        
        return importance_df
        
    except ValueError as e:
        print(f"Error de validación: {e}")
        raise
    except Exception as e:
        raise RuntimeError(f"Error calculando importancia de características: {e}")


def plot_feature_importance(importance_df, top_n=15, save_path=None):
    """
    Genera un gráfico de barras horizontales con la importancia de características.
    
    Args:
        importance_df: DataFrame con columnas 'feature', 'importance_mean', 'importance_std'
        top_n: Número de características principales a mostrar (default: 15)
        save_path: Ruta para guardar la imagen (opcional)
        
    Returns:
        None
    """
    # Tomar las top N características
    df_plot = importance_df.head(top_n).copy()
    
    # Invertir orden para que la más importante quede arriba
    df_plot = df_plot.iloc[::-1]
    
    # Crear figura con tamaño adaptable
    fig_height = max(6, top_n * 0.4)
    plt.figure(figsize=(10, fig_height))
    
    # Crear paleta de colores profesional (degradado)
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(df_plot)))
    
    # Gráfico de barras horizontales
    bars = plt.barh(
        df_plot['feature'], 
        df_plot['importance_mean'],
        xerr=df_plot['importance_std'],
        color=colors,
        edgecolor='navy',
        linewidth=0.5,
        capsize=3,
        error_kw={'elinewidth': 1, 'capthick': 1, 'alpha': 0.7}
    )
    
    # Añadir valores en las barras
    for bar, val in zip(bars, df_plot['importance_mean']):
        plt.text(
            bar.get_width() + 0.002, 
            bar.get_y() + bar.get_height()/2,
            f'{val:.4f}',
            va='center',
            fontsize=9,
            color='dimgray'
        )
    
    # Configuración del gráfico
    plt.xlabel('Importancia (Permutation Importance)', fontsize=12)
    plt.ylabel('Características', fontsize=12)
    plt.title(
        f'Top {top_n} Características más Importantes\nPredicción de Deserción Académica',
        fontsize=14, 
        fontweight='bold'
    )
    
    # Ajustar límites del eje x para dejar espacio a los valores
    x_max = df_plot['importance_mean'].max() + df_plot['importance_std'].max() + 0.02
    plt.xlim(0, x_max)
    
    # Añadir línea vertical en 0
    plt.axvline(x=0, color='gray', linewidth=0.8, linestyle='-')
    
    # Grid suave
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de importancia guardado en: {save_path}")
    
    plt.show()


def explain_prediction(model, student_data, feature_names, threshold=0.5):
    """
    Genera una explicación textual de la predicción para un estudiante.
    
    Args:
        model: Modelo Keras entrenado
        student_data: Array con datos de un estudiante (1D o 2D)
        feature_names: Lista o array con nombres de las características
        threshold: Umbral de clasificación (default: 0.5)
        
    Returns:
        dict: Diccionario con:
            - prediction: Clase predicha (0 o 1)
            - probability: Probabilidad de deserción
            - risk_level: Nivel de riesgo (Bajo, Medio, Alto, Muy Alto)
            - explanation: Explicación textual completa
            - risk_factors: Lista de factores de riesgo principales
            - protective_factors: Lista de factores protectores
    """
    # Asegurar formato correcto (2D para el modelo)
    student_data = np.array(student_data)
    if student_data.ndim == 1:
        student_data = student_data.reshape(1, -1)
    
    # Obtener predicción
    probability = float(model.predict(student_data, verbose=0)[0][0])
    prediction = 1 if probability >= threshold else 0
    
    # Determinar nivel de riesgo
    if probability < 0.25:
        risk_level = "Bajo"
        risk_emoji = "🟢"
    elif probability < 0.50:
        risk_level = "Medio"
        risk_emoji = "🟡"
    elif probability < 0.75:
        risk_level = "Alto"
        risk_emoji = "🟠"
    else:
        risk_level = "Muy Alto"
        risk_emoji = "🔴"
    
    # Analizar valores de características
    # Identificar características con valores extremos (z-score aproximado)
    values = student_data.flatten()
    
    # Crear lista de (característica, valor, contribución potencial)
    feature_analysis = []
    for name, value in zip(feature_names, values):
        feature_analysis.append({
            'name': name,
            'value': value,
            'abs_value': abs(value)  # Para ordenar por magnitud
        })
    
    # Ordenar por valor absoluto (asumiendo datos normalizados)
    feature_analysis.sort(key=lambda x: x['abs_value'], reverse=True)
    
    # Identificar factores de riesgo y protectores
    # (valores positivos altos pueden ser riesgo, negativos pueden ser protectores)
    risk_factors = []
    protective_factors = []
    
    for feat in feature_analysis[:10]:  # Analizar top 10 por magnitud
        if feat['value'] > 0.5:  # Valor alto positivo
            risk_factors.append(f"{feat['name']}: {feat['value']:.2f}")
        elif feat['value'] < -0.5:  # Valor alto negativo
            protective_factors.append(f"{feat['name']}: {feat['value']:.2f}")
    
    # Limitar a los 5 principales de cada tipo
    risk_factors = risk_factors[:5]
    protective_factors = protective_factors[:5]
    
    # Construir explicación textual
    explanation_lines = [
        "=" * 60,
        "ANÁLISIS DE PREDICCIÓN - DESERCIÓN ACADÉMICA",
        "=" * 60,
        "",
        f"📊 RESULTADO DE LA PREDICCIÓN:",
        f"   • Probabilidad de deserción: {probability:.1%}",
        f"   • Clasificación: {'DESERTOR' if prediction == 1 else 'NO DESERTOR'}",
        f"   • Nivel de riesgo: {risk_emoji} {risk_level}",
        ""
    ]
    
    if risk_factors:
        explanation_lines.extend([
            "⚠️  FACTORES DE RIESGO IDENTIFICADOS:",
            *[f"   • {factor}" for factor in risk_factors],
            ""
        ])
    else:
        explanation_lines.extend([
            "⚠️  FACTORES DE RIESGO: No se identificaron factores significativos",
            ""
        ])
    
    if protective_factors:
        explanation_lines.extend([
            "✅ FACTORES PROTECTORES:",
            *[f"   • {factor}" for factor in protective_factors],
            ""
        ])
    else:
        explanation_lines.extend([
            "✅ FACTORES PROTECTORES: No se identificaron factores significativos",
            ""
        ])
    
    # Recomendación basada en el riesgo
    if risk_level in ["Alto", "Muy Alto"]:
        recommendation = "Se recomienda intervención inmediata y seguimiento cercano."
    elif risk_level == "Medio":
        recommendation = "Se sugiere monitoreo preventivo y apoyo académico."
    else:
        recommendation = "Continuar con seguimiento regular."
    
    explanation_lines.extend([
        "💡 RECOMENDACIÓN:",
        f"   {recommendation}",
        "",
        "=" * 60
    ])
    
    explanation = "\n".join(explanation_lines)
    
    # Imprimir explicación
    print(explanation)
    
    return {
        'prediction': prediction,
        'probability': probability,
        'risk_level': risk_level,
        'explanation': explanation,
        'risk_factors': risk_factors,
        'protective_factors': protective_factors
    }


def save_evaluation_report(metrics, plots_paths=None, output_path='evaluation_report.txt', 
                           format='txt', additional_info=None):
    """
    Genera un reporte de evaluación en formato TXT o PDF.
    
    Args:
        metrics: Diccionario con métricas del modelo
            Esperado: {'accuracy': float, 'precision': float, 'recall': float,
                       'f1_score': float, 'roc_auc': float}
        plots_paths: Diccionario con rutas de gráficos generados (opcional)
            Ejemplo: {'confusion_matrix': 'path/cm.png', 'roc_curve': 'path/roc.png'}
        output_path: Ruta del archivo de salida
        format: Formato del reporte ('txt' o 'pdf')
        additional_info: Diccionario con información adicional (opcional)
            Ejemplo: {'model_path': 'models/model.keras', 'n_samples': 1000}
            
    Returns:
        str: Ruta del archivo generado
        
    Raises:
        ValueError: Si el formato no es válido
        ImportError: Si se solicita PDF y fpdf no está instalado
    """
    import os
    from datetime import datetime
    
    if format not in ['txt', 'pdf']:
        raise ValueError(f"Formato '{format}' no válido. Use 'txt' o 'pdf'.")
    
    # Construir contenido del reporte
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_lines = [
        "=" * 70,
        "REPORTE DE EVALUACIÓN - MODELO DE PREDICCIÓN DE DESERCIÓN ACADÉMICA",
        "=" * 70,
        "",
        f"Fecha de generación: {timestamp}",
        ""
    ]
    
    # Información adicional
    if additional_info:
        report_lines.extend([
            "-" * 70,
            "INFORMACIÓN DEL MODELO",
            "-" * 70
        ])
        for key, value in additional_info.items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
    
    # Métricas principales
    report_lines.extend([
        "-" * 70,
        "MÉTRICAS DE EVALUACIÓN",
        "-" * 70
    ])
    
    metric_labels = {
        'accuracy': 'Accuracy (Exactitud)',
        'precision': 'Precision (Precisión)',
        'recall': 'Recall (Sensibilidad)',
        'f1_score': 'F1-Score',
        'roc_auc': 'ROC-AUC'
    }
    
    for key, label in metric_labels.items():
        if key in metrics:
            value = metrics[key]
            if key == 'roc_auc':
                report_lines.append(f"  {label}: {value:.4f}")
            else:
                report_lines.append(f"  {label}: {value:.4f} ({value:.2%})")
    
    report_lines.append("")
    
    # Interpretación de métricas
    report_lines.extend([
        "-" * 70,
        "INTERPRETACIÓN DE RESULTADOS",
        "-" * 70
    ])
    
    # Evaluar desempeño general
    avg_score = np.mean([metrics.get('accuracy', 0), metrics.get('f1_score', 0), 
                         metrics.get('roc_auc', 0)])
    
    if avg_score >= 0.85:
        performance = "EXCELENTE"
        recommendation = "El modelo muestra un desempeño sobresaliente."
    elif avg_score >= 0.75:
        performance = "BUENO"
        recommendation = "El modelo tiene buen desempeño, con margen de mejora."
    elif avg_score >= 0.65:
        performance = "ACEPTABLE"
        recommendation = "Se recomienda ajustar hiperparámetros o aumentar datos."
    else:
        performance = "INSUFICIENTE"
        recommendation = "Se requiere revisión del modelo y los datos de entrenamiento."
    
    report_lines.extend([
        f"  Desempeño general: {performance}",
        f"  Recomendación: {recommendation}",
        ""
    ])
    
    # Análisis de Recall (importante para deserción)
    recall = metrics.get('recall', 0)
    if recall >= 0.80:
        recall_analysis = "El modelo detecta la mayoría de estudiantes en riesgo."
    elif recall >= 0.60:
        recall_analysis = "Algunos estudiantes en riesgo podrían no ser detectados."
    else:
        recall_analysis = "ALERTA: Muchos estudiantes en riesgo no están siendo detectados."
    
    report_lines.extend([
        f"  Análisis de Recall: {recall_analysis}",
        ""
    ])
    
    # Gráficos generados
    if plots_paths:
        report_lines.extend([
            "-" * 70,
            "VISUALIZACIONES GENERADAS",
            "-" * 70
        ])
        for name, path in plots_paths.items():
            report_lines.append(f"  - {name}: {path}")
        report_lines.append("")
    
    # Pie de página
    report_lines.extend([
        "=" * 70,
        "Fin del Reporte",
        "Sistema de Predicción de Deserción Académica",
        "=" * 70
    ])
    
    report_content = "\n".join(report_lines)
    
    # Guardar según formato
    if format == 'txt':
        # Asegurar extensión correcta
        if not output_path.endswith('.txt'):
            output_path = output_path.rsplit('.', 1)[0] + '.txt'
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Reporte TXT guardado en: {output_path}")
        
    elif format == 'pdf':
        try:
            from fpdf import FPDF
        except ImportError:
            raise ImportError(
                "Para generar PDF instale fpdf: pip install fpdf2\n"
                "Alternativamente, use format='txt'"
            )
        
        # Asegurar extensión correcta
        if not output_path.endswith('.pdf'):
            output_path = output_path.rsplit('.', 1)[0] + '.pdf'
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Crear PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Título
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'REPORTE DE EVALUACIÓN', ln=True, align='C')
        pdf.cell(0, 10, 'Modelo de Predicción de Deserción Académica', ln=True, align='C')
        pdf.ln(5)
        
        # Fecha
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, f'Fecha: {timestamp}', ln=True)
        pdf.ln(5)
        
        # Métricas
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'MÉTRICAS DE EVALUACIÓN', ln=True)
        pdf.set_font('Arial', '', 10)
        
        for key, label in metric_labels.items():
            if key in metrics:
                value = metrics[key]
                pdf.cell(0, 6, f'  {label}: {value:.4f}', ln=True)
        
        pdf.ln(5)
        
        # Interpretación
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'INTERPRETACIÓN', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f'  Desempeño: {performance}', ln=True)
        pdf.multi_cell(0, 6, f'  {recommendation}')
        pdf.multi_cell(0, 6, f'  Recall: {recall_analysis}')
        
        # Gráficos (si existen y son accesibles)
        if plots_paths:
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'VISUALIZACIONES', ln=True)
            pdf.set_font('Arial', '', 10)
            
            for name, path in plots_paths.items():
                if os.path.exists(path):
                    try:
                        pdf.ln(3)
                        pdf.cell(0, 6, f'{name}:', ln=True)
                        pdf.image(path, x=10, w=190)
                        pdf.ln(5)
                    except Exception as e:
                        pdf.cell(0, 6, f'  - {name}: {path} (no se pudo incluir)', ln=True)
                else:
                    pdf.cell(0, 6, f'  - {name}: {path}', ln=True)
        
        # Guardar PDF
        pdf.output(output_path)
        print(f"✅ Reporte PDF guardado en: {output_path}")
    
    return output_path


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def run_full_evaluation(model_path, X_test_path, y_test_path, output_dir=None):
    """
    Ejecuta la evaluación completa del modelo.
    
    Args:
        model_path: Ruta al modelo .keras
        X_test_path: Ruta a X_test.npy
        y_test_path: Ruta a y_test.npy
        output_dir: Directorio para guardar gráficos (opcional)
        
    Returns:
        dict: Métricas de evaluación
    """
    import os
    from tensorflow.keras.models import load_model
    
    # Cargar modelo y datos
    print("Cargando modelo y datos...")
    model = load_model(model_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    # Evaluar modelo
    metrics = evaluate_model(model, X_test, y_test)
    
    # Obtener predicciones para gráficos
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    
    # Configurar rutas de guardado
    cm_path = None
    roc_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        roc_path = os.path.join(output_dir, 'roc_curve.png')
    
    # Generar gráficos
    plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
    plot_roc_curve(y_test, y_pred_proba, save_path=roc_path)
    
    return metrics


if __name__ == "__main__":
    # Ejemplo de uso completo
    import os
    from tensorflow.keras.models import load_model
    
    # Rutas por defecto del proyecto
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.keras')
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'plots')
    
    # Cargar datos y modelo
    print("Cargando modelo y datos...")
    model = load_model(MODEL_PATH)
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    feature_names = np.load(os.path.join(DATA_DIR, 'feature_names.npy'), allow_pickle=True)
    
    # 1. Ejecutar evaluación de métricas
    print("\n" + "="*60)
    print("PARTE 1: EVALUACIÓN DEL MODELO")
    print("="*60)
    metrics = evaluate_model(model, X_test, y_test)
    
    # 2. Generar gráficos de evaluación
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    
    plot_confusion_matrix(y_test, y_pred, 
                          save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plot_roc_curve(y_test, y_pred_proba,
                   save_path=os.path.join(OUTPUT_DIR, 'roc_curve.png'))
    
    # 3. Calcular importancia de características
    print("\n" + "="*60)
    print("PARTE 2: EXPLICABILIDAD - IMPORTANCIA DE CARACTERÍSTICAS")
    print("="*60)
    importance_df = get_feature_importance(model, X_test, y_test, feature_names)
    plot_feature_importance(importance_df, top_n=15,
                            save_path=os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    
    # 4. Explicar predicción de un estudiante aleatorio
    print("\n" + "="*60)
    print("PARTE 3: EXPLICABILIDAD - PREDICCIÓN INDIVIDUAL")
    print("="*60)
    random_idx = np.random.randint(0, len(X_test))
    student_explanation = explain_prediction(
        model, 
        X_test[random_idx], 
        feature_names,
        threshold=0.5
    )
    
    print("\n✅ Evaluación completa finalizada.")
    print(f"📁 Gráficos guardados en: {OUTPUT_DIR}")
