import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

st.set_page_config(
    page_title="Predicción Estudiante",
    layout="centered"
)

st.title("Predicción aprobación estudiante")
st.write(
    "Esta aplicación carga un Pipeline registrado en MLflow. "
    "Por eso se ingresan las columnas originales del dataset, no las variables encodeadas."
)

# Conexión al servidor MLflow
mlflow.set_tracking_uri("http://127.0.0.1:9090")

# Cambie la versión si MLflow registra una nueva versión del modelo:
# models:/clase06/1, models:/logisticRegression01/1, etc.
MODEL_URI = "models:/logisticRegression01/10"

@st.cache_resource
def cargar_modelo():
    return mlflow.sklearn.load_model(MODEL_URI)

model = cargar_modelo()

st.sidebar.header("Configuración")
st.sidebar.write(f"Modelo cargado: `{MODEL_URI}`")

st.subheader("Datos del Estudiante")

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad", min_value=0, max_value=100, value=20)
    promedio = st.number_input("Promedio", min_value=0, max_value=10, value=7)
    asistencia = st.number_input("Asistencia", min_value=0, max_value=100, value=75)
    carrera = st.selectbox("Carrera", ["Computacion", "Derecho", "Economia","Medicina","Arquitectura","Industrial"])
    modalidad = st.selectbox("Modalidad", ["Presencial", "Virtual", "Hibrida"])

datos = pd.DataFrame([{
    "carrera": carrera,
    "modalidad": modalidad,
    "edad": edad,
    "promedio": promedio,
    "asistencias": asistencia,
}])

st.subheader("Datos enviados al modelo")
st.dataframe(datos)

if st.button("Predecir"):
    prediccion = model.predict(datos)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(datos)[0]
        prob_no = proba[0]
        prob_si = proba[1]
    else:
        prob_no = None
        prob_si = None

    if prediccion == 1:
        st.success("Predicción: el estudiante SÍ podría aprobar.")
    else:
        st.warning("Predicción: el estudiante NO aprobara.")


st.caption(
    "Nota: el notebook eliminó la columna 'beca'. Por eso esta aplicación tampoco la solicita."
)
