"""
Módulo de utilidades para visualización en Streamlit.
Funciones adaptadas para mostrar métricas, gráficos y explicaciones
del modelo de predicción de deserción académica.

Uso:
    from src.streamlit_utils import (
        display_metrics_streamlit,
        display_evaluation_plots,
        display_prediction_explanation
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# Importar streamlit solo cuando se use (evita errores en otros contextos)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def display_metrics_streamlit(metrics_dict):
    """
    Muestra métricas de evaluación en columnas de Streamlit usando st.metric().
    
    Args:
        metrics_dict: Diccionario con métricas del modelo
            Esperado: {'accuracy': float, 'precision': float, 'recall': float, 
                       'f1_score': float, 'roc_auc': float}
    
    Example:
        metrics = evaluate_model(model, X_test, y_test)
        display_metrics_streamlit(metrics)
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit no está instalado. Instalar con: pip install streamlit")
    
    st.subheader("📊 Métricas de Evaluación del Modelo")
    
    # Crear 5 columnas para las métricas principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Mapeo de nombres para display
    metric_config = {
        'accuracy': {'label': 'Accuracy', 'icon': '🎯', 'format': '{:.2%}'},
        'precision': {'label': 'Precision', 'icon': '✅', 'format': '{:.2%}'},
        'recall': {'label': 'Recall', 'icon': '🔍', 'format': '{:.2%}'},
        'f1_score': {'label': 'F1-Score', 'icon': '⚖️', 'format': '{:.2%}'},
        'roc_auc': {'label': 'ROC-AUC', 'icon': '📈', 'format': '{:.4f}'}
    }
    
    columns = [col1, col2, col3, col4, col5]
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    for col, key in zip(columns, metrics_keys):
        if key in metrics_dict:
            config = metric_config[key]
            value = metrics_dict[key]
            
            with col:
                st.metric(
                    label=f"{config['icon']} {config['label']}",
                    value=config['format'].format(value)
                )
    
    # Interpretación de métricas
    with st.expander("ℹ️ Interpretación de Métricas"):
        st.markdown("""
        - **Accuracy**: Proporción de predicciones correctas sobre el total
        - **Precision**: De los predichos como desertores, ¿cuántos lo son realmente?
        - **Recall**: De los desertores reales, ¿cuántos fueron detectados?
        - **F1-Score**: Balance entre Precision y Recall
        - **ROC-AUC**: Capacidad discriminativa del modelo (1.0 = perfecto, 0.5 = aleatorio)
        """)


def display_evaluation_plots(model, X_test, y_test, show_all=True):
    """
    Genera y muestra gráficos de evaluación directamente en Streamlit.
    
    Args:
        model: Modelo Keras entrenado
        X_test: Datos de prueba (features)
        y_test: Etiquetas reales
        show_all: Si True, muestra todos los gráficos. Si False, permite selección
        
    Returns:
        dict: Diccionario con las figuras generadas (para uso adicional)
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit no está instalado. Instalar con: pip install streamlit")
    
    # Obtener predicciones
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_true = np.array(y_test).flatten()
    
    st.subheader("📈 Visualizaciones de Evaluación")
    
    figures = {}
    
    if show_all:
        tabs = st.tabs(["Matriz de Confusión", "Curva ROC", "Precision-Recall", "Distribución"])
    else:
        plot_option = st.selectbox(
            "Seleccionar visualización:",
            ["Matriz de Confusión", "Curva ROC", "Precision-Recall", "Distribución"]
        )
        tabs = [st.container()]
    
    # ============== MATRIZ DE CONFUSIÓN ==============
    tab_idx = 0 if show_all else (0 if plot_option == "Matriz de Confusión" else -1)
    if show_all or plot_option == "Matriz de Confusión":
        with tabs[tab_idx] if show_all else tabs[0]:
            fig_cm, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Valores absolutos
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Deserta', 'Deserta'],
                        yticklabels=['No Deserta', 'Deserta'],
                        annot_kws={'size': 14}, ax=axes[0])
            axes[0].set_title('Valores Absolutos', fontweight='bold')
            axes[0].set_ylabel('Real')
            axes[0].set_xlabel('Predicho')
            
            # Porcentajes
            cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                        xticklabels=['No Deserta', 'Deserta'],
                        yticklabels=['No Deserta', 'Deserta'],
                        annot_kws={'size': 14}, ax=axes[1])
            axes[1].set_title('Porcentajes', fontweight='bold')
            axes[1].set_ylabel('Real')
            axes[1].set_xlabel('Predicho')
            
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)
            figures['confusion_matrix'] = fig_cm
            
            # Métricas de la matriz
            tn, fp, fn, tp = cm.ravel()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("TN (Correctos No Desertores)", tn)
            col2.metric("FP (Falsos Desertores)", fp)
            col3.metric("FN (No Detectados) ⚠️", fn)
            col4.metric("TP (Desertores Detectados)", tp)
    
    # ============== CURVA ROC ==============
    tab_idx = 1 if show_all else (0 if plot_option == "Curva ROC" else -1)
    if show_all or plot_option == "Curva ROC":
        with tabs[tab_idx] if show_all else tabs[0]:
            fig_roc, ax = plt.subplots(figsize=(10, 7))
            
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            
            # Curva ROC
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC (AUC = {roc_auc:.4f})')
            ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
            
            # Línea diagonal
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Aleatorio (AUC = 0.50)')
            
            # Punto óptimo
            optimal_idx = np.argmax(tpr - fpr)
            ax.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o',
                       color='red', s=150, label=f'Óptimo (umbral={thresholds[optimal_idx]:.3f})')
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('Tasa de Falsos Positivos')
            ax.set_ylabel('Tasa de Verdaderos Positivos')
            ax.set_title('Curva ROC', fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig_roc)
            plt.close(fig_roc)
            figures['roc_curve'] = fig_roc
    
    # ============== PRECISION-RECALL ==============
    tab_idx = 2 if show_all else (0 if plot_option == "Precision-Recall" else -1)
    if show_all or plot_option == "Precision-Recall":
        with tabs[tab_idx] if show_all else tabs[0]:
            fig_pr, ax = plt.subplots(figsize=(10, 7))
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_prec = average_precision_score(y_true, y_pred_proba)
            baseline = np.sum(y_true) / len(y_true)
            
            ax.plot(recall, precision, color='darkgreen', lw=2,
                    label=f'PR (AP = {avg_prec:.4f})')
            ax.fill_between(recall, precision, alpha=0.3, color='darkgreen')
            ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                       label=f'Baseline = {baseline:.3f}')
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Curva Precision-Recall', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig_pr)
            plt.close(fig_pr)
            figures['pr_curve'] = fig_pr
    
    # ============== DISTRIBUCIÓN DE PROBABILIDADES ==============
    tab_idx = 3 if show_all else (0 if plot_option == "Distribución" else -1)
    if show_all or plot_option == "Distribución":
        with tabs[tab_idx] if show_all else tabs[0]:
            fig_dist, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histograma
            axes[0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7,
                         label='No Desertores', color='green')
            axes[0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7,
                         label='Desertores', color='red')
            axes[0].axvline(x=0.5, color='black', linestyle='--', lw=2)
            axes[0].set_xlabel('Probabilidad')
            axes[0].set_ylabel('Frecuencia')
            axes[0].set_title('Distribución de Probabilidades', fontweight='bold')
            axes[0].legend()
            
            # Box plot
            data_box = [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]]
            bp = axes[1].boxplot(data_box, labels=['No Desertores', 'Desertores'],
                                  patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[1].set_ylabel('Probabilidad')
            axes[1].set_title('Box Plot por Clase', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close(fig_dist)
            figures['distribution'] = fig_dist
    
    return figures


def display_prediction_explanation(model, student_data, feature_names, threshold=0.5):
    """
    Muestra explicación de predicción individual en Streamlit.
    
    Args:
        model: Modelo Keras entrenado
        student_data: Array con datos de un estudiante
        feature_names: Lista de nombres de características
        threshold: Umbral de clasificación (default: 0.5)
        
    Returns:
        dict: Resultado de la predicción con detalles
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit no está instalado. Instalar con: pip install streamlit")
    
    # Preparar datos
    student_data = np.array(student_data)
    if student_data.ndim == 1:
        student_data = student_data.reshape(1, -1)
    
    # Predicción
    probability = float(model.predict(student_data, verbose=0)[0][0])
    prediction = 1 if probability >= threshold else 0
    
    # Determinar nivel de riesgo
    if probability < 0.25:
        risk_level = "Bajo"
        risk_color = "green"
        st_func = st.success
    elif probability < 0.50:
        risk_level = "Medio"
        risk_color = "yellow"
        st_func = st.info
    elif probability < 0.75:
        risk_level = "Alto"
        risk_color = "orange"
        st_func = st.warning
    else:
        risk_level = "Muy Alto"
        risk_color = "red"
        st_func = st.error
    
    # Mostrar resultado principal
    st.subheader("🎓 Análisis de Predicción Individual")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Probabilidad de Deserción",
            value=f"{probability:.1%}"
        )
    
    with col2:
        st.metric(
            label="Clasificación",
            value="DESERTOR" if prediction == 1 else "NO DESERTOR"
        )
    
    with col3:
        st.metric(
            label="Nivel de Riesgo",
            value=risk_level
        )
    
    # Mostrar mensaje según riesgo
    if risk_level == "Bajo":
        st.success("🟢 **Riesgo Bajo**: El estudiante tiene baja probabilidad de desertar. "
                   "Continuar con seguimiento regular.")
    elif risk_level == "Medio":
        st.info("🟡 **Riesgo Medio**: Se recomienda monitoreo preventivo y apoyo académico.")
    elif risk_level == "Alto":
        st.warning("🟠 **Riesgo Alto**: Se requiere atención prioritaria. "
                   "Considerar intervención académica y/o psicológica.")
    else:
        st.error("🔴 **Riesgo Muy Alto**: ¡Intervención urgente necesaria! "
                 "Contactar al estudiante de inmediato.")
    
    # Análisis de características
    values = student_data.flatten()
    feature_df = pd.DataFrame({
        'Característica': feature_names,
        'Valor': values,
        'Valor Absoluto': np.abs(values)
    }).sort_values('Valor Absoluto', ascending=False)
    
    # Identificar factores de riesgo y protectores
    risk_factors = feature_df[feature_df['Valor'] > 0.5].head(5)
    protective_factors = feature_df[feature_df['Valor'] < -0.5].head(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚠️ Factores de Riesgo**")
        if len(risk_factors) > 0:
            for _, row in risk_factors.iterrows():
                st.markdown(f"- {row['Característica']}: `{row['Valor']:.2f}`")
        else:
            st.markdown("_No se identificaron factores significativos_")
    
    with col2:
        st.markdown("**✅ Factores Protectores**")
        if len(protective_factors) > 0:
            for _, row in protective_factors.iterrows():
                st.markdown(f"- {row['Característica']}: `{row['Valor']:.2f}`")
        else:
            st.markdown("_No se identificaron factores significativos_")
    
    # Gráfico de características principales
    with st.expander("📊 Ver todas las características"):
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.25)))
        
        # Top 15 por valor absoluto
        top_features = feature_df.head(15).iloc[::-1]
        colors = ['red' if v > 0 else 'green' for v in top_features['Valor']]
        
        ax.barh(top_features['Característica'], top_features['Valor'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Valor (normalizado)')
        ax.set_title('Top 15 Características por Magnitud')
        
        st.pyplot(fig)
        plt.close(fig)
    
    return {
        'prediction': prediction,
        'probability': probability,
        'risk_level': risk_level,
        'risk_factors': risk_factors['Característica'].tolist() if len(risk_factors) > 0 else [],
        'protective_factors': protective_factors['Característica'].tolist() if len(protective_factors) > 0 else []
    }


def display_feature_importance_streamlit(importance_df, top_n=15):
    """
    Muestra gráfico de importancia de características en Streamlit.
    
    Args:
        importance_df: DataFrame con columnas 'feature', 'importance_mean', 'importance_std'
        top_n: Número de características a mostrar
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit no está instalado")
    
    st.subheader("🔍 Importancia de Características")
    
    df_plot = importance_df.head(top_n).iloc[::-1]
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(df_plot)))
    
    bars = ax.barh(df_plot['feature'], df_plot['importance_mean'],
                   xerr=df_plot['importance_std'] if 'importance_std' in df_plot.columns else None,
                   color=colors, edgecolor='navy', linewidth=0.5, capsize=3)
    
    for bar, val in zip(bars, df_plot['importance_mean']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    ax.set_xlabel('Importancia (Permutation Importance)')
    ax.set_title(f'Top {top_n} Características más Importantes', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)
    plt.close(fig)
    
    # Tabla con valores
    with st.expander("📋 Ver tabla completa"):
        st.dataframe(
            importance_df.head(top_n).style.format({
                'importance_mean': '{:.4f}',
                'importance_std': '{:.4f}' if 'importance_std' in importance_df.columns else '{}'
            }),
            use_container_width=True
        )


# Función auxiliar para crear página de evaluación completa
def create_evaluation_page(model, X_test, y_test, feature_names, importance_df=None):
    """
    Crea una página completa de evaluación en Streamlit.
    
    Args:
        model: Modelo Keras entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        feature_names: Nombres de características
        importance_df: DataFrame de importancia (opcional, se calcula si no se provee)
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit no está instalado")
    
    st.title("🎓 Evaluación del Modelo de Deserción Académica")
    
    # Importar función de evaluación
    from src.evaluation_explainability import evaluate_model, get_feature_importance
    
    # Sidebar para navegación
    section = st.sidebar.radio(
        "Secciones",
        ["📊 Métricas", "📈 Visualizaciones", "🔍 Importancia", "🎯 Predicción Individual"]
    )
    
    if section == "📊 Métricas":
        metrics = evaluate_model(model, X_test, y_test)
        display_metrics_streamlit(metrics)
        
    elif section == "📈 Visualizaciones":
        display_evaluation_plots(model, X_test, y_test)
        
    elif section == "🔍 Importancia":
        if importance_df is None:
            with st.spinner("Calculando importancia de características..."):
                importance_df = get_feature_importance(model, X_test, y_test, feature_names)
        display_feature_importance_streamlit(importance_df)
        
    elif section == "🎯 Predicción Individual":
        st.subheader("Seleccionar Estudiante")
        idx = st.number_input("Índice del estudiante", 0, len(X_test)-1, 0)
        if st.button("Analizar Estudiante"):
            display_prediction_explanation(model, X_test[idx], feature_names)
