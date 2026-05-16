# Task Ml Flow

1. Instalar las librerías del proyecto, mediante el archivo requiments.txt 

### Ejecutar MlFlow

1. Ejecutar el servidor de mlflow, desde la carpeta raíz
* opción 1: mlflow server --backend-store-uri sqlite:///mlflow.db  --host 0.0.0.0 --port 9090
* opción 2 - problemas con cors : mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 9090 --allowed-hosts "*" --cors-allowed-origins "*"


### Para entrenar y crear el modelo utiliza jupiter lab
1. En un nuevo terminal
2. Desde la carpeta raíz, ejecutar jupyter-lab [taskLab.ipynb](taskLab.ipynb)

**Importante:** Antes de la API o Streamlit, entrena y registra el modelo en MLflow con `taskLab.ipynb`. La carpeta `mlartifacts/` no se sube al repo; debes generarla localmente al entrenar.
### Use model from API
1. En un nuevo terminal
2. Desde la carpeta raíz, ejecutar python  [api_flask.py](api_flask.py)

### Use model from Streamlit
1. En un nuevo terminal
2. Desde la carpeta raíz, ejecutar streamlit run  [app_streamlit.py](app_streamlit.py)

**_Importante_**: Tanto para el uso del api o del streamlit se debe validar el modelo y el url de MlFlow.

`
mlflow.set_tracking_uri("http://127.0.0.1:9090")
MODEL_URI = "models:/logisticRegression01/1"
mlflow.sklearn.load_model(MODEL_URI)
`

`
mlflow.set_tracking_uri("http://127.0.0.1:9090")
model = mlflow.sklearn.load_model("models:/logisticRegression01/1")
`