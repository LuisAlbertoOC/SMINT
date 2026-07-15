import base64
import io
import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps


app = Flask(__name__)


RUTA_MODELO = os.path.join(
    "modelos",
    "modelo_digitos_16x16.json"
)


def cargar_modelo_json(ruta):
    with open(ruta, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


modelo_json = cargar_modelo_json(RUTA_MODELO)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def predecir_numpy(imagen_16x16):
    """
    Recibe una imagen normalizada de forma (16, 16, 1)
    y aplica las capas Dense exportadas en JSON.
    """

    salida = imagen_16x16.reshape(1, -1)

    for capa in modelo_json["layers"]:
        pesos = np.array(capa["weights"], dtype=np.float32)
        bias = np.array(capa["bias"], dtype=np.float32)

        salida = np.dot(salida, pesos) + bias

        activacion = capa["activation"]

        if activacion == "relu":
            salida = relu(salida)

        elif activacion == "softmax":
            salida = softmax(salida.reshape(-1)).reshape(1, -1)

        elif activacion == "linear":
            salida = salida

        else:
            raise ValueError(f"Activación no soportada: {activacion}")

    probabilidades = salida.reshape(-1)
    prediccion = int(np.argmax(probabilidades))

    return prediccion, probabilidades


def convertir_data_url_a_imagen(data_url):
    """
    Convierte la imagen enviada desde el canvas HTML a imagen PIL.
    """

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    imagen_bytes = base64.b64decode(data_url)
    imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGBA")

    fondo = Image.new("RGBA", imagen.size, (255, 255, 255, 255))
    fondo.alpha_composite(imagen)

    return fondo.convert("L")


def preprocesar_imagen(imagen):
    """
    Convierte el dibujo del usuario al formato del modelo:
    - Escala de grises
    - Inversión de colores: fondo negro y número claro
    - Recorte del área dibujada
    - Centrado
    - Redimensionado a 16x16
    - Normalización 0-255 a 0-1
    """

    # El canvas dibuja negro sobre blanco.
    # MNIST usa número claro sobre fondo oscuro.
    imagen = ImageOps.invert(imagen)

    arreglo = np.array(imagen)

    # Detectar pixeles donde hay tinta.
    coords = np.argwhere(arreglo > 25)

    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    imagen_recortada = imagen.crop(
        (
            x_min,
            y_min,
            x_max + 1,
            y_max + 1
        )
    )

    ancho, alto = imagen_recortada.size
    lado = max(ancho, alto)

    margen = int(lado * 0.35)
    lado_con_margen = lado + margen * 2

    lienzo = Image.new(
        "L",
        (lado_con_margen, lado_con_margen),
        0
    )

    x = (lado_con_margen - ancho) // 2
    y = (lado_con_margen - alto) // 2

    lienzo.paste(imagen_recortada, (x, y))

    try:
        filtro = Image.Resampling.LANCZOS
    except AttributeError:
        filtro = Image.LANCZOS

    imagen_16 = lienzo.resize(
        (16, 16),
        filtro
    )

    arreglo_16 = np.array(imagen_16).astype("float32") / 255.0
    arreglo_16 = arreglo_16.reshape(16, 16, 1)

    return arreglo_16


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        datos = request.get_json()

        if datos is None or "imagen" not in datos:
            return jsonify({
                "error": "No se recibió ninguna imagen."
            }), 400

        imagen = convertir_data_url_a_imagen(datos["imagen"])
        imagen_procesada = preprocesar_imagen(imagen)

        if imagen_procesada is None:
            return jsonify({
                "error": "No se detectó ningún número dibujado."
            }), 400

        prediccion, probabilidades = predecir_numpy(imagen_procesada)

        probabilidades_lista = [
            {
                "digito": i,
                "probabilidad": round(float(probabilidades[i]), 6),
                "porcentaje": round(float(probabilidades[i]) * 100, 2)
            }
            for i in range(10)
        ]

        return jsonify({
            "prediccion": prediccion,
            "probabilidades": probabilidades_lista
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
