from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# CREACIÓN DE LA API
# ============================================================

app = FastAPI(
    title="API de Predicción de Combustible",
    description="API para predecir el tipo de combustible de vehículos",
    version="1.0.0"
)


# ============================================================
# CARGAR MODELO Y PREPROCESAMIENTO
# ============================================================

artefactos = joblib.load("modelo_vehiculos.joblib")

modelo = artefactos["modelo"]
frecuencias = artefactos["frecuencias"]
encoder = artefactos["encoder"]
label_encoder = artefactos["label_encoder"]

variables_frecuencia = artefactos["variables_frecuencia"]
variables_onehot = artefactos["variables_onehot"]
variables_numericas = artefactos["variables_numericas"]


# ============================================================
# MODELO DE DATOS QUE RECIBIRÁ LA API
# ============================================================

class Vehiculo(BaseModel):

    tipo_transaccion: str
    marca: str
    modelo: str
    pais: str
    clase: str
    tipo: str
    tipo_servicio: str
    canton: str
    color_1: str
    color_2: str
    persona_natural_juridica: str

    ano_modelo: int
    avaluo: float
    cilindraje: float
    ano_proceso: int
    dias_compra_proceso: float
    diferencia_anio_modelo: float

    modelo_futuro: str
    compra_posterior_proceso: str


# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================

@app.get("/")
def inicio():

    return {
        "mensaje": "API de predicción de combustible funcionando",
        "modelo": "Árbol de Decisión",
        "caracteristicas": modelo.n_features_in_
    }


# ============================================================
# ENDPOINT DE PREDICCIÓN
# ============================================================

@app.post("/predecir")
def predecir(vehiculo: Vehiculo):

    # --------------------------------------------------------
    # Convertir los datos recibidos a DataFrame
    # --------------------------------------------------------

    datos = {
        "TIPO TRANSACCIÓN": vehiculo.tipo_transaccion,
        "MARCA": vehiculo.marca,
        "MODELO": vehiculo.modelo,
        "PAÍS": vehiculo.pais,
        "CLASE": vehiculo.clase,
        "TIPO": vehiculo.tipo,
        "TIPO SERVICIO": vehiculo.tipo_servicio,
        "CANTÓN": vehiculo.canton,
        "COLOR 1": vehiculo.color_1,
        "COLOR 2": vehiculo.color_2,
        "PERSONA NATURAL - JURÍDICA": vehiculo.persona_natural_juridica,

        "AÑO MODELO": vehiculo.ano_modelo,
        "AVALÚO": vehiculo.avaluo,
        "CILINDRAJE": vehiculo.cilindraje,
        "AÑO PROCESO": vehiculo.ano_proceso,
        "DIAS_COMPRA_PROCESO": vehiculo.dias_compra_proceso,
        "DIFERENCIA_ANIO_MODELO": vehiculo.diferencia_anio_modelo,

        "MODELO_FUTURO": vehiculo.modelo_futuro,
        "COMPRA_POSTERIOR_PROCESO": vehiculo.compra_posterior_proceso
    }

    df = pd.DataFrame([datos])


    # --------------------------------------------------------
    # FREQUENCY ENCODING
    # --------------------------------------------------------

    for columna in variables_frecuencia:

        frecuencia = frecuencias[columna]

        df[columna] = (
            df[columna]
            .map(frecuencia)
            .fillna(0)
        )


    # --------------------------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------------------------

    datos_onehot = encoder.transform(
        df[variables_onehot]
    )

    nombres_onehot = encoder.get_feature_names_out(
        variables_onehot
    )

    df_onehot = pd.DataFrame(
        datos_onehot,
        columns=nombres_onehot
    )


    # --------------------------------------------------------
    # VARIABLES NUMÉRICAS
    # --------------------------------------------------------

    df_numericas = df[variables_numericas].copy()


    # --------------------------------------------------------
    # CONSTRUIR LAS 102 VARIABLES FINALES
    # --------------------------------------------------------

    X_final = pd.concat(
        [
            df_numericas.reset_index(drop=True),
            df[variables_frecuencia].reset_index(drop=True),
            df_onehot.reset_index(drop=True)
        ],
        axis=1
    )


    # --------------------------------------------------------
    # ASEGURAR EL MISMO ORDEN DE VARIABLES DEL ENTRENAMIENTO
    # --------------------------------------------------------

    if X_final.shape[1] != modelo.n_features_in_:

        return {
            "error": "La cantidad de características no coincide",
            "caracteristicas_generadas": X_final.shape[1],
            "caracteristicas_esperadas": modelo.n_features_in_
        }


    # --------------------------------------------------------
    # REALIZAR PREDICCIÓN
    # --------------------------------------------------------

    prediccion = modelo.predict(X_final)


    # --------------------------------------------------------
    # CONVERTIR CÓDIGO A NOMBRE DEL COMBUSTIBLE
    # --------------------------------------------------------

    combustible = label_encoder.inverse_transform(
        prediccion
    )[0]


    # --------------------------------------------------------
    # RESPUESTA DE LA API
    # --------------------------------------------------------

    return {
        "prediccion": combustible,
        "modelo_utilizado": "Árbol de Decisión",
        "caracteristicas_utilizadas": X_final.shape[1]
    }