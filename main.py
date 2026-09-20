from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

# Obtiene la carpeta donde se encuentra este archivo main.py
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# CREACIÓN DE LA API
# ============================================================

app = FastAPI(
    title="API de Predicción de Combustible",
    description="API para predecir el tipo de combustible de vehículos",
    version="1.0.0"
)


# ============================================================
# CONFIGURACIÓN DE LA INTERFAZ WEB
# ============================================================

# Busca la carpeta templates ubicada junto a main.py
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# ============================================================
# CARGAR MODELO Y PREPROCESAMIENTO
# ============================================================

# Carga el modelo y los elementos de preprocesamiento
artefactos = joblib.load(
    BASE_DIR / "modelo_vehiculos.joblib"
)

# Recuperar los componentes guardados
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

    # --------------------------------------------------------
    # VARIABLES CATEGÓRICAS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VARIABLES NUMÉRICAS
    # --------------------------------------------------------

    ano_modelo: int
    avaluo: float
    cilindraje: float
    ano_proceso: int
    dias_compra_proceso: float
    diferencia_anio_modelo: float

    # --------------------------------------------------------
    # VARIABLES DERIVADAS
    # --------------------------------------------------------

    modelo_futuro: str
    compra_posterior_proceso: str


# ============================================================
# PÁGINA PRINCIPAL - FORMULARIO WEB
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def inicio(request: Request):

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request
        }
    )


# ============================================================
# ENDPOINT DE INFORMACIÓN DE LA API
# ============================================================

@app.get("/api-info")
def api_info():

    return {
        "mensaje": "API de predicción de combustible funcionando",
        "modelo": "Árbol de Decisión",
        "caracteristicas": modelo.n_features_in_,
        "endpoint_prediccion": "/predecir",
        "documentacion": "/docs"
    }


# ============================================================
# ENDPOINT DE PREDICCIÓN
# ============================================================

@app.post("/predecir")
def predecir(vehiculo: Vehiculo):

    # --------------------------------------------------------
    # CONVERTIR LOS DATOS RECIBIDOS A DATAFRAME
    # --------------------------------------------------------

    datos = {

        "TIPO TRANSACCIÓN":
            vehiculo.tipo_transaccion,

        "MARCA":
            vehiculo.marca,

        "MODELO":
            vehiculo.modelo,

        "PAÍS":
            vehiculo.pais,

        "CLASE":
            vehiculo.clase,

        "TIPO":
            vehiculo.tipo,

        "TIPO SERVICIO":
            vehiculo.tipo_servicio,

        "CANTÓN":
            vehiculo.canton,

        "COLOR 1":
            vehiculo.color_1,

        "COLOR 2":
            vehiculo.color_2,

        "PERSONA NATURAL - JURÍDICA":
            vehiculo.persona_natural_juridica,

        "AÑO MODELO":
            vehiculo.ano_modelo,

        "AVALÚO":
            vehiculo.avaluo,

        "CILINDRAJE":
            vehiculo.cilindraje,

        "AÑO PROCESO":
            vehiculo.ano_proceso,

        "DIAS_COMPRA_PROCESO":
            vehiculo.dias_compra_proceso,

        "DIFERENCIA_ANIO_MODELO":
            vehiculo.diferencia_anio_modelo,

        "MODELO_FUTURO":
            vehiculo.modelo_futuro,

        "COMPRA_POSTERIOR_PROCESO":
            vehiculo.compra_posterior_proceso
    }

    # Crear DataFrame con un solo vehículo
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

    df_numericas = df[
        variables_numericas
    ].copy()


    # --------------------------------------------------------
    # CONSTRUIR LAS 102 VARIABLES FINALES
    # --------------------------------------------------------

    X_final = pd.concat(
        [

            # 6 variables numéricas
            df_numericas.reset_index(drop=True),

            # 3 variables con Frequency Encoding
            df[variables_frecuencia]
            .reset_index(drop=True),

            # 93 variables One-Hot
            df_onehot
            .reset_index(drop=True)

        ],
        axis=1
    )


    # --------------------------------------------------------
    # VERIFICAR CANTIDAD DE CARACTERÍSTICAS
    # --------------------------------------------------------

    if X_final.shape[1] != modelo.n_features_in_:

        return {

            "error":
                "La cantidad de características no coincide",

            "caracteristicas_generadas":
                X_final.shape[1],

            "caracteristicas_esperadas":
                modelo.n_features_in_
        }


    # --------------------------------------------------------
    # REALIZAR PREDICCIÓN
    # --------------------------------------------------------

    prediccion = modelo.predict(
        X_final
    )


    # --------------------------------------------------------
    # CONVERTIR CÓDIGO A NOMBRE DEL COMBUSTIBLE
    # --------------------------------------------------------

    combustible = label_encoder.inverse_transform(
        prediccion
    )[0]


    # --------------------------------------------------------
    # OBTENER PROBABILIDADES
    # --------------------------------------------------------

    probabilidades = modelo.predict_proba(
        X_final
    )[0]

    nombres_clases = label_encoder.classes_

    probabilidades_dict = {

        clase:
            round(float(probabilidad) * 100, 2)

        for clase, probabilidad
        in zip(
            nombres_clases,
            probabilidades
        )
    }


    # --------------------------------------------------------
    # RESPUESTA DE LA API
    # --------------------------------------------------------

    return {

        "prediccion":
            combustible,

        "modelo_utilizado":
            "Árbol de Decisión",

        "caracteristicas_utilizadas":
            X_final.shape[1],

        "probabilidades":
            probabilidades_dict
    }