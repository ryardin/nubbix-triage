"""Configuración del servicio: credenciales y parámetros de los modelos.

Las API keys se leen del archivo .env (que NUNCA se sube al repositorio).
Los parámetros de los modelos están acá, en un solo lugar, con la justificación
de cada valor.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Raíz del proyecto, calculada desde la ubicación de este archivo (src/config.py).
# Así las rutas funcionan en cualquier máquina, sin depender de C:\dev\... ni de
# la carpeta desde la que se ejecute el programa.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Carga las variables del .env como variables de entorno.
# Si una variable ya existe en el sistema, se respeta la del sistema.
load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(Exception):
    """Error de configuración: falta una API key, proveedor desconocido, etc."""


# --------------------------------------------------------------------------- #
# Proveedores disponibles
# --------------------------------------------------------------------------- #
# Para cada proveedor: qué variable de entorno tiene su API key y qué modelo usamos.
# Elegimos en cada uno un modelo chico y rápido: la tarea (clasificar un ticket y
# proponer una respuesta corta) no necesita un modelo grande, y así el costo y la
# latencia se mantienen bajos. Esta elección se valida después con los evals.
#
# max_tokens es el máximo de tokens de SALIDA (razonamiento interno + JSON) por llamada. Es un
# tope: se pagan los tokens generados, no el máximo. Va por proveedor porque la restricción es
# solo de Groq. Medido el 2026-11-25 con v1 y v2 y 3 tickets (claro, vago y mixto):
# - Gemini y Groq usaron entre 52 y 113 tokens de salida.
# - Un JSON del tamaño máximo (answer de 500 caracteres + 3 acciones) ocupó 137 tokens en
#   Gemini, 135 en Groq y 246 en Anthropic.
PROVIDERS = {
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        # flash-lite no usa razonamiento interno: en las pruebas, flash (con razonamiento)
        # gastó ~10 veces más tokens y tardó bastante más para preguntas simples.
        "model": "gemini-3.5-flash-lite",
        "max_tokens": 2048,
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        # qwen3.8-27b respondió sin razonamiento interno y en menos de 1 segundo.
        "model": "qwen/qwen3.8-27b",
        # El nivel gratuito limita la salida a 1000 tokens por minuto: con 1024 respondía 429
        # ("Limit 1000, Requested 1024") en parte de los intentos.
        "max_tokens": 512,
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        # El modelo más chico y barato de Claude (USD 0,10 / 0,50 por millón de tokens).
        "model": "claude-haiku-5-5",
        # Razona antes de responder y eso también consume max_tokens: con 512, las 3 llamadas
        # con el ticket mixto se cortaron sin completar el JSON. Con 2048 usó 757.
        "max_tokens": 2048,
    },
}


# --------------------------------------------------------------------------- #
# Parámetros comunes a todos los proveedores
# --------------------------------------------------------------------------- #
# Son los mismos para los tres proveedores, para que la comparación en los evals sea justa.
# (max_tokens es la excepción: está en PROVIDERS, por proveedor.)

# temperature = 0: queremos respuestas consistentes (la misma consulta debería dar la
# misma categoría siempre), no creatividad. Anthropic no acepta este parámetro.
TEMPERATURE = 0.0

# timeout = 20 s por llamada: la más lenta de las pruebas tardó ~17 s, y fue de un modelo con
# razonamiento que no usamos (los elegidos tardaron 8 s o menos). Si un proveedor no responde
# en 20 s, conviene pasar al siguiente en lugar de seguir esperando.
TIMEOUT_S = 20.0

# max_retries = 1: ante un error transitorio se reintenta una sola vez por proveedor.
# Reintentar más suma espera para el usuario, y hay otros proveedores para responder.
MAX_RETRIES = 1


# --------------------------------------------------------------------------- #
# Cambio de proveedor (failover)
# --------------------------------------------------------------------------- #
# Orden en que se prueban los proveedores si uno falla: primero los gratuitos, y el pago
# (Anthropic) como último recurso. El primero es el proveedor por defecto.
# Los evals NO usan esta lista: miden cada proveedor por separado.
PROVIDER_ORDER = ["gemini", "groq", "anthropic"]


# --------------------------------------------------------------------------- #
# Funciones de acceso
# --------------------------------------------------------------------------- #
def get_api_key(provider_name: str) -> str:
    """Devuelve la API key del proveedor, o un error claro si no está configurada."""
    env_var = PROVIDERS[provider_name]["api_key_env"]
    api_key = os.getenv(env_var, "").strip()
    if not api_key:
        raise ConfigError(
            f"Falta la variable {env_var} para usar el proveedor '{provider_name}'. "
            f"Copiá .env.example como .env y completá tu API key."
        )
    return api_key
