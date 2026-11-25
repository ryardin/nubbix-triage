"""Proveedores de LLM detrás de una interfaz común (ver base.py).

El resto del sistema obtiene un proveedor solo con su nombre:

    from src.providers import get_provider
    provider = get_provider("groq")
    response = provider.generate(system, messages)
"""

from src import config
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import LLMProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.groq_provider import GroqProvider

# Qué clase corresponde a cada nombre de proveedor.
PROVIDER_CLASSES = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(name: str) -> LLMProvider:
    """Crea el proveedor indicado, ya configurado con su API key y sus parámetros."""
    if name not in PROVIDER_CLASSES:
        options = ", ".join(PROVIDER_CLASSES)
        raise config.ConfigError(f"Proveedor desconocido: '{name}'. Opciones válidas: {options}.")

    provider_class = PROVIDER_CLASSES[name]
    return provider_class(
        api_key=config.get_api_key(name),
        model=config.PROVIDERS[name]["model"],
        temperature=config.TEMPERATURE,
        max_tokens=config.PROVIDERS[name]["max_tokens"],
        timeout_s=config.TIMEOUT_S,
        max_retries=config.MAX_RETRIES,
    )
