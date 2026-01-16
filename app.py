"""
================================================================================
APLICACIÓN STREAMLIT - PREDA
Sistema de Predicción de Deserción Académica con ANN
Responsable: Takeshy
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_data, preprocess_data
from src.model_ann import load_trained_model, predict_single_student
from src.evaluation_explainability import (
    evaluate_model,
    get_feature_importance,
    explain_prediction
)
from src.streamlit_utils import (
    display_metrics_streamlit,
    display_evaluation_plots,
    display_prediction_explanation,
    display_feature_importance_streamlit
)

st.set_page_config(
    page_title="PREDA - Predicción de Deserción Académica",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


MARITAL_STATUS_MAP = {
    1: "Soltero(a)", 2: "Casado(a)", 3: "Viudo(a)",
    4: "Divorciado(a)", 5: "Unión de Hecho", 6: "Separado(a)"
}

NATIONALITY_MAP = {
    1: "Perú", 2: "Alemania", 6: "España", 11: "Italia",
    13: "Países Bajos", 14: "Reino Unido", 17: "Lituania", 21: "Angola",
    22: "Cabo Verde", 24: "Guinea-Bissau", 25: "Mozambique", 26: "Santo Tomé y Príncipe",
    32: "Turquía", 41: "Brasil", 62: "Rumania", 100: "Moldavia",
    101: "México", 103: "Ucrania", 105: "Rusia", 108: "Cuba", 109: "Colombia"
}

COURSE_MAP = {
    33: "Biotecnología", 171: "Diseño", 8014: "Educación",
    9003: "Animación y Diseño Multimedia", 9070: "Trabajo Social",
    9085: "Veterinaria", 9119: "Informática", 9130: "Turismo",
    9147: "Enfermería", 9238: "Higiene Oral", 9254: "Publicidad y Marketing",
    9500: "Periodismo", 9556: "Gestión", 9670: "Administración",
    9773: "Ingeniería Civil", 9853: "Comunicación Empresarial", 9991: "Agronomía"
}

APPLICATION_MODE_MAP = {
    1: "Admisión general", 2: "Modalidad especial",
    5: "Admisión especial (región)", 7: "Titulares de otros cursos superiores",
    10: "Modalidad ordinaria", 15: "Estudiante internacional",
    16: "Admisión especial (extraordinaria)", 17: "Segunda convocatoria",
    18: "Tercera convocatoria", 26: "Traslado interno",
    27: "Traslado externo", 39: "Mayores de 23 años",
    42: "Transferencia", 43: "Cambio de carrera", 44: "Egresados de institutos técnicos",
    51: "Cambio de institución/carrera", 53: "Programa corto", 57: "Cambio de institución (internacional)"
}

PREVIOUS_QUALIFICATION_MAP = {
    1: "Educación secundaria completa", 2: "Educación superior - Licenciatura/Título",
    3: "Educación superior - Maestría", 4: "Educación superior - Doctorado",
    5: "Estudios superiores en curso", 6: "5to año secundaria - no completado",
    9: "4to año secundaria - no completado", 10: "3er año secundaria",
    12: "3er año secundaria - no completado", 14: "Carrera técnica básica",
    19: "Cambio de carrera en universidad", 38: "Instituto técnico superior",
    39: "Estudios en instituto técnico", 40: "5to año secundaria - no completado",
    42: "Carrera técnica profesional", 43: "Estudios en carrera técnica"
}

QUALIFICATION_MAP = {
    1: "Educación secundaria completa (5to año)", 2: "Educación superior - Licenciatura/Título",
    3: "Educación superior - Maestría", 4: "Educación superior - Doctorado",
    5: "Estudios superiores en curso", 6: "5to año secundaria - no completado",
    9: "4to año secundaria - no completado", 10: "3er año secundaria",
    11: "Secundaria completa (5to año)", 12: "Primaria completa (6to grado)",
    13: "Primaria 4to grado", 14: "Sin educación formal",
    18: "Secundaria incompleta", 19: "Primaria incompleta", 22: "Primaria inicial incompleta",
    26: "Desconocido", 27: "No completó secundaria", 29: "Primaria completa (6to grado)",
    30: "Primaria 6to grado", 34: "Sin educación formal",
    35: "Alfabetizado sin escolaridad completa", 36: "Educación básica alternativa",
    37: "Desconocido", 38: "Secundaria incompleta", 39: "Primaria incompleta",
    40: "Secundaria completa", 41: "Desconocido", 42: "3er año secundaria - no completado",
    43: "4to año secundaria - no completado", 44: "Educación secundaria completa"
}

OCCUPATION_MAP = {
    0: "Estudiante", 1: "Representantes del poder legislativo",
    2: "Especialistas intelectuales y científicos", 3: "Técnicos de nivel intermedio",
    4: "Personal administrativo", 5: "Trabajadores de servicios y vendedores",
    6: "Agricultores y trabajadores agropecuarios", 7: "Trabajadores calificados de industria",
    8: "Operadores de máquinas y ensambladores", 9: "Trabajadores no calificados",
    10: "Profesiones de las Fuerzas Armadas", 90: "Otros / Sin información",
    99: "Trabajador por cuenta propia", 122: "Profesional de salud",
    123: "Profesor", 125: "Especialista en TI", 131: "Técnico intermedio",
    135: "Técnico de TI", 143: "Servicios de atención personal",
    144: "Servicios de protección y seguridad"
}


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 2.5rem 0;
        margin-bottom: 2rem;
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        color: #1a202c;
    }
    
    .main-header .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #2d3748;
        margin: 2.5rem 0 1.5rem 0;
        padding-left: 1.5rem;
        border-left: 5px solid #667eea;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 2rem 0;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.1);
    }
    
    .info-box h3 {
        color: #2c5282;
        margin-top: 0;
        font-weight: 700;
    }
    
    .risk-card {
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .risk-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
    }
    
    .risk-low::before {
        background: linear-gradient(90deg, #38a169 0%, #48bb78 100%);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fffaf0 0%, #feebc8 100%);
    }
    
    .risk-medium::before {
        background: linear-gradient(90deg, #ed8936 0%, #f6ad55 100%);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
    }
    
    .risk-high::before {
        background: linear-gradient(90deg, #e53e3e 0%, #fc8181 100%);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        font-size: 1.15rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    .form-section {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    
    .form-section h3 {
        color: #2d3748;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #667eea;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
    }
    
    hr {
        margin: 3rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model = load_trained_model('models/best_model.keras')
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_training_history():
    try:
        with open('models/best_model_history.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_dataset_sample():
    try:
        return load_data('data/raw/students_dropout.csv')
    except:
        return None

def create_gauge_chart(probability, risk_level):
    color = "#38a169" if risk_level == "BAJO" else "#ed8936" if risk_level == "MEDIO" else "#e53e3e"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        title = {'text': "<b>Probabilidad de Deserción</b>", 'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 50, 'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(56, 161, 105, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(237, 137, 54, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(229, 62, 62, 0.2)'}
            ],
            'threshold': {'line': {'color': color, 'width': 6}, 'thickness': 0.85, 'value': probability * 100}
        }
    ))
    
    fig.update_layout(height=450, margin=dict(l=30, r=30, t=80, b=30), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def create_risk_card(resultado):
    prob = resultado['probabilidad_desercion']
    nivel = resultado['nivel_riesgo']
    
    if nivel == "BAJO":
        card_class, icon, color = "risk-low", "✓", "#38a169"
        mensaje = "El estudiante muestra excelentes perspectivas de permanencia."
        recomendaciones = [
            "✓ Mantener apoyo académico actual",
            "✓ Fomentar actividades extracurriculares",
            "✓ Seguimiento periódico preventivo"
        ]
    elif nivel == "MEDIO":
        card_class, icon, color = "risk-medium", "⚠", "#ed8936"
        mensaje = "El estudiante presenta indicadores de riesgo que requieren atención."
        recomendaciones = [
            "→ Asignar tutor académico personalizado",
            "→ Ofrecer asesorías en materias difíciles",
            "→ Evaluar situación socioeconómica",
            "→ Seguimiento mensual de rendimiento"
        ]
    else:
        card_class, icon, color = "risk-high", "!", "#e53e3e"
        mensaje = "Alto riesgo de deserción. Intervención inmediata requerida."
        recomendaciones = [
            "⚠ URGENTE: Contactar estudiante y familia",
            "⚠ Tutor personal y acompañamiento intensivo",
            "⚠ Evaluar apoyos económicos urgentes",
            "⚠ Consejería psicopedagógica",
            "⚠ Seguimiento semanal obligatorio"
        ]
    
    recs_html = "".join([f"<li style='margin: 0.5rem 0;'>{r}</li>" for r in recomendaciones])
    
    return f"""
    <div class="risk-card {card_class}">
        <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
            <div style="width: 60px; height: 60px; background: {color}; border-radius: 50%;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 2rem; color: white; font-weight: bold; margin-right: 1.5rem;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);">{icon}</div>
            <div>
                <h2 style="margin: 0; color: {color}; font-size: 1.8rem; font-weight: 800;">
                    Riesgo {nivel}
                </h2>
                <p style="margin: 0.5rem 0 0 0; color: #4a5568; font-size: 1.3rem; font-weight: 600;">
                    Probabilidad: {prob*100:.1f}%
                </p>
            </div>
        </div>
        <p style="font-size: 1.1rem; line-height: 1.7; color: #2d3748; margin: 1.5rem 0;">{mensaje}</p>
        <h4 style="margin: 2rem 0 1rem 0; color: #2d3748; font-size: 1.2rem; font-weight: 700;
                   padding-bottom: 0.5rem; border-bottom: 2px solid {color};">
            📋 Plan de Acción
        </h4>
        <ul style="font-size: 1rem; color: #4a5568; line-height: 2; padding-left: 1.5rem;">
            {recs_html}
        </ul>
    </div>
    """


def page_prediction():
    st.markdown('<p class="main-header"><span class="gradient-text">PREDA</span><br>Sistema de Predicción de Deserción Académica</p>', unsafe_allow_html=True)
    
    model, error = load_model()
    if model is None:
        st.error(f"❌ Error al cargar el modelo: {error}")
        return
    
    st.success("✅ Modelo cargado correctamente")
    
    st.markdown("""
    <div class="info-box">
        <h3>📝 Instrucciones</h3>
        <p>Complete el formulario con la información del estudiante. Todos los campos usan valores descriptivos
        basados en el dataset UCI de deserción académica.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        
        # SECCIÓN 1: DATOS PERSONALES
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 👤 Datos Personales")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Edad al inscribirse", 16, 70, 20)
            gender = st.selectbox("Género", [1, 0], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
            marital_status = st.selectbox("Estado Civil", list(MARITAL_STATUS_MAP.keys()), 
                                         format_func=lambda x: MARITAL_STATUS_MAP[x])
        
        with col2:
            nacionality = st.selectbox("Nacionalidad", list(NATIONALITY_MAP.keys()), 
                                      format_func=lambda x: NATIONALITY_MAP[x])
            international = st.selectbox("¿Estudiante internacional?", [0, 1], 
                                        format_func=lambda x: "Sí" if x == 1 else "No")
            displaced = st.selectbox("¿Estudiante desplazado?", [0, 1], 
                                    format_func=lambda x: "Sí" if x == 1 else "No")
        
        with col3:
            special_needs = st.selectbox("¿Necesidades educativas especiales?", [0, 1], 
                                        format_func=lambda x: "Sí" if x == 1 else "No")
            scholarship = st.selectbox("¿Becario?", [0, 1], 
                                      format_func=lambda x: "Sí" if x == 1 else "No")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # SECCIÓN 2: INFORMACIÓN ACADÉMICA
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 🎓 Información Académica")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            course = st.selectbox("Carrera", list(COURSE_MAP.keys()), 
                                 format_func=lambda x: COURSE_MAP[x], index=6)
            application_mode = st.selectbox("Modo de aplicación", list(APPLICATION_MODE_MAP.keys()), 
                                           format_func=lambda x: APPLICATION_MODE_MAP[x], index=0)
        
        with col2:
            application_order = st.number_input("Orden de aplicación (0-9)", 0, 9, 1,
                                               help="Orden de preferencia en la solicitud")
            daytime_evening = st.selectbox("Horario de asistencia", [1, 0], 
                                          format_func=lambda x: "Diurno" if x == 1 else "Nocturno")
        
        with col3:
            prev_qualification = st.selectbox("Calificación previa", list(PREVIOUS_QUALIFICATION_MAP.keys()), 
                                             format_func=lambda x: PREVIOUS_QUALIFICATION_MAP[x], index=0)
            prev_qualification_grade = st.number_input("Nota de calificación previa (0-200)", 0.0, 200.0, 120.0)
        
        admission_grade = st.number_input("Nota de admisión (0-200)", 0.0, 200.0, 120.0,
                                         help="Calificación con la que ingresó a la institución")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # SECCIÓN 3: INFORMACIÓN FAMILIAR
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 👨‍👩‍👧‍👦 Información Familiar")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Madre**")
            mothers_qualification = st.selectbox("Nivel educativo de la madre", 
                                                list(QUALIFICATION_MAP.keys()), 
                                                format_func=lambda x: QUALIFICATION_MAP[x], index=0)
            mothers_occupation = st.selectbox("Ocupación de la madre", 
                                             list(OCCUPATION_MAP.keys()), 
                                             format_func=lambda x: OCCUPATION_MAP[x], index=0)
        
        with col2:
            st.markdown("**Padre**")
            fathers_qualification = st.selectbox("Nivel educativo del padre", 
                                                list(QUALIFICATION_MAP.keys()), 
                                                format_func=lambda x: QUALIFICATION_MAP[x], index=0)
            fathers_occupation = st.selectbox("Ocupación del padre", 
                                             list(OCCUPATION_MAP.keys()), 
                                             format_func=lambda x: OCCUPATION_MAP[x], index=0)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # SECCIÓN 4: SITUACIÓN FINANCIERA
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 💰 Situación Financiera")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            debtor = st.selectbox("¿Es deudor?", [0, 1], 
                                 format_func=lambda x: "Sí" if x == 1 else "No")
        
        with col2:
            tuition_up_to_date = st.selectbox("¿Matrícula al día?", [1, 0], 
                                             format_func=lambda x: "Sí" if x == 1 else "No")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # SECCIÓN 5 Y 6: RENDIMIENTO ACADÉMICO
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Rendimiento Académico")
        st.info("📌 El modelo evalúa estudiantes que han completado al menos dos semestres académicos")
        
        tab1, tab2 = st.tabs(["Primer Semestre", "Segundo Semestre"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                cu_1st_credited = st.number_input("Asignaturas acreditadas", 0, 30, 0, key="1c",
                                                 help="Asignaturas reconocidas por experiencia previa")
                cu_1st_enrolled = st.number_input("Asignaturas inscritas", 0, 30, 6, key="1e",
                                                 help="Total de asignaturas matriculadas")
            with col2:
                cu_1st_evaluations = st.number_input("Evaluaciones realizadas", 0, 50, 6, key="1ev",
                                                    help="Número total de evaluaciones presentadas")
                cu_1st_approved = st.number_input("Asignaturas aprobadas", 0, 30, 5, key="1a",
                                                 help="Asignaturas aprobadas exitosamente")
            with col3:
                cu_1st_grade = st.number_input("Nota promedio (0-20)", 0.0, 20.0, 13.0, key="1g",
                                              help="Promedio de calificaciones del semestre")
                cu_1st_without_eval = st.number_input("Sin evaluaciones", 0, 30, 0, key="1w",
                                                     help="Asignaturas sin evaluaciones presentadas")
        
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                cu_2nd_credited = st.number_input("Asignaturas acreditadas", 0, 30, 0, key="2c",
                                                 help="Asignaturas reconocidas por experiencia previa")
                cu_2nd_enrolled = st.number_input("Asignaturas inscritas", 0, 30, 6, key="2e",
                                                 help="Total de asignaturas matriculadas")
            with col2:
                cu_2nd_evaluations = st.number_input("Evaluaciones realizadas", 0, 50, 6, key="2ev",
                                                    help="Número total de evaluaciones presentadas")
                cu_2nd_approved = st.number_input("Asignaturas aprobadas", 0, 30, 5, key="2a",
                                                 help="Asignaturas aprobadas exitosamente")
            with col3:
                cu_2nd_grade = st.number_input("Nota promedio (0-20)", 0.0, 20.0, 13.0, key="2g",
                                              help="Promedio de calificaciones del semestre")
                cu_2nd_without_eval = st.number_input("Sin evaluaciones", 0, 30, 0, key="2w",
                                                     help="Asignaturas sin evaluaciones presentadas")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # SECCIÓN 7: INDICADORES MACROECONÓMICOS
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 📈 Indicadores Macroeconómicos")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            unemployment_rate = st.number_input("Tasa de desempleo (%)", 0.0, 30.0, 10.0)
        with col2:
            inflation_rate = st.number_input("Tasa de inflación (%)", -5.0, 20.0, 1.5)
        with col3:
            gdp = st.number_input("PIB (crecimiento %)", -10.0, 10.0, 1.0)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # BOTÓN
        submitted = st.form_submit_button("🔮 Generar Predicción", use_container_width=True)
    
    # PROCESAR PREDICCIÓN
    if submitted:
        with st.spinner("🧠 Analizando datos del estudiante..."):
            student_data = np.array([[
                marital_status, application_mode, application_order, course,
                daytime_evening, prev_qualification, prev_qualification_grade,
                nacionality, mothers_qualification, fathers_qualification,
                mothers_occupation, fathers_occupation, admission_grade,
                displaced, special_needs, debtor, tuition_up_to_date,
                gender, scholarship, age, international,
                cu_1st_credited, cu_1st_enrolled, cu_1st_evaluations, cu_1st_approved,
                cu_1st_grade, cu_1st_without_eval,
                cu_2nd_credited, cu_2nd_enrolled, cu_2nd_evaluations, cu_2nd_approved,
                cu_2nd_grade, cu_2nd_without_eval,
                unemployment_rate, inflation_rate, gdp
            ]])
            
            # Normalizar
            from sklearn.preprocessing import StandardScaler
            try:
                import pickle
                with open('data/processed/scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                student_data_scaled = scaler.transform(student_data)
            except:
                student_data_scaled = student_data
            
            resultado = predict_single_student(model, student_data_scaled)
            
            # MOSTRAR RESULTADOS
            st.markdown('<p class="sub-header">📊 Resultados de la Predicción</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                gauge_fig = create_gauge_chart(resultado['probabilidad_desercion'], resultado['nivel_riesgo'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Renderizar la tarjeta de riesgo estilizada
                st.markdown(create_risk_card(resultado), unsafe_allow_html=True)
            
            # Métricas adicionales en tarjetas pequeñas
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 10px;">
                    <small>Nivel de Riesgo</small>
                    <h3 style="margin:0; color: #2d3748;">{resultado['nivel_riesgo']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
            with m2:
                delta = (resultado['probabilidad_desercion'] * 100) - 50
                color_delta = "red" if delta > 0 else "green"
                symbol = "↑" if delta > 0 else "↓"
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 10px;">
                    <small>vs. Promedio</small>
                    <h3 style="margin:0; color: {color_delta};">{symbol} {abs(delta):.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                # Simulación de confianza (basada en cuán lejos está del 0.5)
                confianza = 50 + (abs(resultado['probabilidad_desercion'] - 0.5) * 100)
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 10px;">
                    <small>Confianza del Modelo</small>
                    <h3 style="margin:0; color: #2d3748;">{confianza:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
            with m4:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 10px;">
                    <small>Predicción</small>
                    <h3 style="margin:0; color: #2d3748;">{resultado['prediccion_texto']}</h3>
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# PÁGINA: DESCRIPCIÓN DEL PROYECTO
# =============================================================================

def page_about():
    st.markdown('<p class="main-header">Acerca del Proyecto</p>', unsafe_allow_html=True)
    
    # Contenedor principal con efecto cristal
    st.markdown("""
    <div class="info-box">
        <h3>🚀 Objetivo de PREDA</h3>
        <p><strong>PREDA (Predicción de Deserción Académica)</strong> es una herramienta analítica avanzada diseñada 
        para instituciones educativas. Utiliza Inteligencia Artificial para detectar patrones tempranos de abandono 
        estudiantil, permitiendo una intervención proactiva y personalizada.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="form-section" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("### 🧠 Arquitectura del Modelo")
        st.markdown("""
        El sistema utiliza una **Red Neuronal Artificial (ANN)** optimizada con TensorFlow/Keras:
        
        * **Entrada:** 36 variables (Demográficas, Socioeconómicas, Académicas).
        * **Capas Ocultas:** Estructura densa (64 → 32 → 16 neuronas) con activación ReLU.
        * **Regularización:** Capas de *Dropout* (30%) y *Batch Normalization* para evitar sobreajuste.
        * **Salida:** Activación Sigmoid para probabilidad binaria (0-1).
        """)
        
        # Diagrama simple con código (mermaid style) o imagen si existiera
        st.code("""
Input (36) → Dense(64) → Dropout → Dense(32) → Dense(16) → Output(1)
        """, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="form-section" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("### 📊 Variables Clave")
        st.markdown("""
        El modelo analiza múltiples dimensiones del estudiante, mapeadas automáticamente en el sistema:
        
        * **Factores Académicos:** Notas de admisión, historial de evaluaciones, unidades aprobadas.
        * **Entorno Socioeconómico:** Ocupación y educación de los padres, estado de becas, deudas.
        * **Factores Externos:** Tasa de desempleo, inflación y PIB del país.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💻 Equipo de Desarrollo")
    
    # Tarjetas de equipo minimalistas
    c1, c2, c3, c4, c5 = st.columns(5)
    
    team = [
        ("Roberto", "Coordinador"), ("David", "Data Scientist"), 
        ("Jorge", "ML Engineer"), ("Takeshy", "Frontend/Streamlit"), 
        ("Adrián", "QA & Metrics")
    ]
    
    cols = [c1, c2, c3, c4, c5]
    
    for col, (name, role) in zip(cols, team):
        with col:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <div style="font-size: 2rem;">👤</div>
                <div style="font-weight: bold; margin-top: 5px;">{name}</div>
                <div style="font-size: 0.8rem; color: gray;">{role}</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PÁGINA: ANÁLISIS Y MÉTRICAS
# =============================================================================

def page_analysis():
    st.markdown('<p class="main-header">Análisis y Métricas</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Métricas del Modelo", "📊 Exploración de Datos", "🎯 Evaluación del Modelo", "🔍 Explicabilidad"])
    
    with tab1:
        history = load_training_history()
        
        if history:
            st.markdown('<div class="info-box"><h3>Rendimiento del Entrenamiento</h3></div>', unsafe_allow_html=True)
            
            # Métricas principales
            hm1, hm2, hm3, hm4 = st.columns(4)
            hist = history['history']
            
            with hm1: st.metric("Accuracy Final", f"{hist['val_accuracy'][-1]:.4f}")
            with hm2: st.metric("Precision", f"{hist['val_precision'][-1]:.4f}")
            with hm3: st.metric("Recall", f"{hist['val_recall'][-1]:.4f}")
            with hm4: st.metric("Loss", f"{hist['val_loss'][-1]:.4f}")
            
            st.markdown("### 📈 Curvas de Aprendizaje")
            
            # Gráficos con Plotly
            epochs = list(range(1, len(hist['loss']) + 1))
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Pérdida (Loss)", "Precisión (Accuracy)"))
            
            # Loss
            fig.add_trace(go.Scatter(x=epochs, y=hist['loss'], name='Train Loss', line=dict(color='#667eea')), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=hist['val_loss'], name='Val Loss', line=dict(color='#e53e3e')), row=1, col=1)
            
            # Accuracy
            fig.add_trace(go.Scatter(x=epochs, y=hist['accuracy'], name='Train Acc', line=dict(color='#667eea')), row=1, col=2)
            fig.add_trace(go.Scatter(x=epochs, y=hist['val_accuracy'], name='Val Acc', line=dict(color='#38a169')), row=1, col=2)
            
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ No se encontró el historial de entrenamiento (models/best_model_history.json).")

    with tab2:
        df = load_dataset_sample()
        if df is not None:
            st.markdown('<div class="info-box"><h3>Distribución del Dataset</h3></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Distribución de Edad")
                fig_age = px.histogram(df, x='Age_at_enrollment', color='Target', 
                                     color_discrete_map={'Dropout': '#e53e3e', 'Graduate': '#38a169', 'Enrolled': '#ed8936'},
                                     nbins=30)
                fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_age, use_container_width=True)
                
            with col2:
                st.markdown("#### Estado Civil vs Deserción")
                # Crear un gráfico de barras simple agrupado
                fig_bar = px.histogram(df, x='Marital_status', color='Target', barmode='group',
                                     color_discrete_map={'Dropout': '#e53e3e', 'Graduate': '#38a169', 'Enrolled': '#ed8936'})
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.error("No se pudo cargar el dataset de muestra.")
    
    # =========================================================================
    # TAB 3: EVALUACIÓN DEL MODELO (NUEVO)
    # =========================================================================
    with tab3:
        st.markdown('<div class="info-box"><h3>🎯 Evaluación del Modelo en Datos de Prueba</h3></div>', unsafe_allow_html=True)
        
        # Cargar datos y modelo
        try:
            model = load_trained_model()
            X_test = np.load('data/processed/X_test.npy')
            y_test = np.load('data/processed/y_test.npy')
            feature_names = np.load('data/processed/feature_names.npy', allow_pickle=True)
            
            st.success(f"✅ Datos cargados: {len(X_test)} muestras de prueba")
            
            # Calcular y mostrar métricas
            with st.spinner("Calculando métricas de evaluación..."):
                metrics = evaluate_model(model, X_test, y_test)
            
            # Mostrar métricas con la función de streamlit_utils
            display_metrics_streamlit(metrics)
            
            st.markdown("---")
            
            # Mostrar visualizaciones
            st.markdown("### 📈 Visualizaciones de Evaluación")
            display_evaluation_plots(model, X_test, y_test)
            
        except FileNotFoundError as e:
            st.error(f"⚠️ No se encontraron los archivos necesarios: {e}")
            st.info("Asegúrate de que existan los archivos en data/processed/: X_test.npy, y_test.npy, feature_names.npy")
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
    
    # =========================================================================
    # TAB 4: EXPLICABILIDAD (NUEVO)
    # =========================================================================
    with tab4:
        st.markdown('<div class="info-box"><h3>🔍 Explicabilidad e Importancia de Características</h3></div>', unsafe_allow_html=True)
        
        try:
            model = load_trained_model()
            X_test = np.load('data/processed/X_test.npy')
            y_test = np.load('data/processed/y_test.npy')
            feature_names = np.load('data/processed/feature_names.npy', allow_pickle=True)
            
            # Sub-tabs para explicabilidad
            exp_tab1, exp_tab2 = st.tabs(["📊 Importancia de Características", "🎓 Análisis Individual"])
            
            with exp_tab1:
                st.markdown("#### Importancia de Características (Permutation Importance)")
                st.info("Este análisis muestra qué características tienen mayor impacto en las predicciones del modelo.")
                
                if st.button("🔄 Calcular Importancia", key="calc_importance"):
                    with st.spinner("Calculando importancia de características... (puede tomar unos minutos)"):
                        importance_df = get_feature_importance(model, X_test, y_test, feature_names, n_repeats=5)
                        st.session_state['importance_df'] = importance_df
                
                if 'importance_df' in st.session_state:
                    display_feature_importance_streamlit(st.session_state['importance_df'], top_n=15)
            
            with exp_tab2:
                st.markdown("#### Análisis de Predicción Individual")
                st.info("Selecciona un estudiante del conjunto de prueba para ver el análisis detallado de su predicción.")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    student_idx = st.number_input(
                        "Índice del estudiante",
                        min_value=0,
                        max_value=len(X_test)-1,
                        value=0,
                        help=f"Selecciona un valor entre 0 y {len(X_test)-1}"
                    )
                    
                    real_label = "Desertó" if y_test[student_idx] == 1 else "No desertó"
                    st.metric("Etiqueta Real", real_label)
                    
                    if st.button("🔍 Analizar Estudiante", key="analyze_student"):
                        st.session_state['analyze_idx'] = student_idx
                
                with col2:
                    if 'analyze_idx' in st.session_state:
                        display_prediction_explanation(
                            model, 
                            X_test[st.session_state['analyze_idx']], 
                            feature_names
                        )
                        
        except FileNotFoundError as e:
            st.error(f"⚠️ No se encontraron los archivos necesarios: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

# =============================================================================
# MAIN NAVIGATION
# =============================================================================

def main():
    with st.sidebar:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/2995/2995620.png", width=100) 
        st.markdown("<h2 style='text-align: center;'> PREDA</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Sistema Inteligente de Predicción</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        menu = st.radio(
            "Navegación",
            ["🔮 Predicción", "📊 Análisis y Métricas", "ℹ️ Acerca del Proyecto"],
            index=0
        )
        
        st.markdown("---")
        st.info("Version 1.0.0 \n\n Desplegado con Streamlit")

    # Routing
    if menu == "🔮 Predicción":
        page_prediction()
    elif menu == "ℹ️ Acerca del Proyecto":
        page_about()
    elif menu == "📊 Análisis y Métricas":
        page_analysis()

if __name__ == "__main__":
    main()