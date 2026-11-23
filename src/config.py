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
PROVIDERS = {
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        # flash-lite no usa razonamiento interno: en las pruebas, flash (con razonamiento)
        # gastó ~10 veces más tokens y tardó bastante más para preguntas simples.
        "model": "gemini-3.5-flash-lite",
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        # qwen3.8-27b respondió sin razonamiento interno y en menos de 1 segundo.
        "model": "qwen/qwen3.8-27b",
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        # El modelo más chico y barato de Claude (USD 0,10 / 0,50 por millón de tokens).
        "model": "claude-haiku-5-5",
    },
}


# --------------------------------------------------------------------------- #
# Parámetros comunes a todos los proveedores
# --------------------------------------------------------------------------- #
# Son los mismos para los tres proveedores, para que la comparación en los evals sea justa.

# temperature = 0: queremos respuestas consistentes (la misma consulta debería dar la
# misma categoría siempre), no creatividad. Anthropic no acepta este parámetro.
TEMPERATURE = 0.0

# max_tokens = 512: el JSON de respuesta medido ocupa ~60 tokens, y el más largo posible
# (answer de 500 caracteres + 3 acciones) no llega a 250. 512 deja margen para que el JSON
# nunca se corte a la mitad (sería inválido y obligaría a reintentar).
# Empezamos con 1024, pero el nivel gratuito de Groq limita la salida a 1000 tokens por
# minuto y rechazaba las llamadas con 429 antes de generar nada.
MAX_TOKENS = 512

# timeout = 30 s: la llamada más lenta en las pruebas tardó ~17 s (un modelo con
# razonamiento). 30 s da margen sin dejar la CLI colgada si el proveedor no responde.
TIMEOUT_S = 30.0


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


def get_default_provider() -> str:
    """Proveedor a usar cuando no se indica uno (variable DEFAULT_PROVIDER del .env)."""
    return os.getenv("DEFAULT_PROVIDER", "gemini").strip()
