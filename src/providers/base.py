"""Interfaz común para todos los proveedores de LLM.

El resto del sistema (pipeline, evals) solo conoce `LLMProvider` y `LLMResponse`.
Cada proveedor concreto (Gemini, Groq, Anthropic) hereda de `LLMProvider` e
implementa únicamente `_call`, que traduce la interfaz común al SDK de su proveedor.

La política de reintentos ante errores de API vive acá, en un solo lugar, para que
sea idéntica para todos los proveedores y quede registrada (campo `attempts`).
"""

import logging
import random
import time
from abc import ABC, abstractmethod

# Los mensajes de diagnóstico van por logging (stderr), nunca por print (stdout):
# stdout queda reservado para el JSON de salida, que debe ser consumible por otro sistema.
logger = logging.getLogger(__name__)

# Un mensaje de la conversación: {"role": "user" | "assistant", "content": "..."}.
# El system prompt NO va en esta lista: se pasa aparte porque cada SDK lo ubica distinto.
Message = dict[str, str]


# --------------------------------------------------------------------------- #
# Respuesta normalizada
# --------------------------------------------------------------------------- #
class LLMResponse:
    """Respuesta de un LLM, con la misma forma sin importar el proveedor."""

    def __init__(
        self,
        text: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_s: float,
        attempts: int,
    ) -> None:
        self.text = text                            # Texto generado por el modelo (todavía sin validar)
        self.provider = provider                    # "gemini" | "groq" | "anthropic"
        self.model = model                          # Modelo exacto usado, p. ej. "gemini-3.5-flash-lite"
        self.prompt_tokens = prompt_tokens          # Tokens de entrada (system + mensajes)
        self.completion_tokens = completion_tokens  # Tokens de salida facturados (incluye razonamiento interno si lo hay)
        self.total_tokens = total_tokens            # prompt_tokens + completion_tokens
        self.latency_s = latency_s                  # Tiempo total de generate() en segundos, incluyendo reintentos
        self.attempts = attempts                    # Cantidad de llamadas hechas a la API (1 = salió a la primera)

    def __repr__(self) -> str:
        """Representación legible al hacer print() (útil para depurar)."""
        return (
            f"LLMResponse(provider={self.provider!r}, model={self.model!r}, "
            f"prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens}, "
            f"total_tokens={self.total_tokens}, latency_s={self.latency_s}, "
            f"attempts={self.attempts}, text={self.text!r})"
        )


# --------------------------------------------------------------------------- #
# Errores propios
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Error base de cualquier proveedor."""


class TransientLLMError(LLMError):
    """Error temporal: tiene sentido reintentar (429 rate limit, 5xx, timeout, red)."""


class PermanentLLMError(LLMError):
    """Error que no se arregla reintentando (401 key inválida, 400 request inválido, modelo inexistente)."""


# --------------------------------------------------------------------------- #
# Interfaz común
# --------------------------------------------------------------------------- #
class LLMProvider(ABC):
    """Clase base abstracta: define QUÉ hace un proveedor; cada subclase define CÓMO."""

    name: str = "base"  # Cada subclase lo redefine ("gemini", "groq", "anthropic")

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        base_delay_s: float = 1.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries      # Reintentos además del primer intento
        self.base_delay_s = base_delay_s    # Espera inicial del backoff exponencial

    def generate(self, system: str, messages: list[Message]) -> LLMResponse:
        """Llama al modelo con reintentos ante errores transitorios.

        Política: hasta `max_retries` reintentos con backoff exponencial + jitter
        (1 s, 2 s, 4 s... más un azar de hasta 0,5 s). Los errores permanentes se
        propagan de inmediato, sin reintentar.
        """
        start = time.perf_counter()
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                text, prompt_tokens, completion_tokens = self._call(system, messages)
            except TransientLLMError as error:
                if attempt == total_attempts:
                    raise TransientLLMError(
                        f"[{self.name}] falló tras {attempt} intentos: {error}"
                    ) from error
                delay = self.base_delay_s * 2 ** (attempt - 1) + random.uniform(0, 0.5)
                logger.warning(
                    "[%s] error transitorio (intento %d/%d): %s. Reintentando en %.1f s",
                    self.name, attempt, total_attempts, error, delay,
                )
                time.sleep(delay)
                continue

            return LLMResponse(
                text=text,
                provider=self.name,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_s=round(time.perf_counter() - start, 3),
                attempts=attempt,
            )

        raise AssertionError("generate() terminó sin resultado")

    @abstractmethod
    def _call(self, system: str, messages: list[Message]) -> tuple[str, int, int]:
        """Hace UNA llamada a la API del proveedor.

        Debe devolver (texto, prompt_tokens, completion_tokens) y traducir las
        excepciones de su SDK a TransientLLMError o PermanentLLMError.
        """
