from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("http://localhost:9090") # cambiar en función de su servidor
## http://0.0.0.0:9090/#/models/task_model/versions/1
model = mlflow.sklearn.load_model("models:/logisticRegression01/4") # cambiar en función de su modelo

app = Flask(__name__)

@app.route("/predecir/dos", methods=["GET"])
def predecir_02():

    edad = int(request.args.get("edad"))
    promedio = int(request.args.get("promedio"))
    asistencias = int(request.args.get("asistencias"))

    carrera = request.args.get("carrera")
    modalidad = request.args.get("modalidad")

    if not edad or not promedio or not asistencias or not carrera or not modalidad:
        return jsonify({"error": "No enviaste data"})
    datos = pd.DataFrame([{
        "carrera": carrera,
        "modalidad": modalidad,
        "edad": edad,
        "promedio": promedio,
        "asistencias": asistencias,
    }])
    prediccion = model.predict(datos)[0]
    pred = "Si" if prediccion == 1 else "No"

    return jsonify({"promedio": promedio, "asistencia": asistencias,"aprobado":pred})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
