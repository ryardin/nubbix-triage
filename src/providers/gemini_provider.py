"""Adaptador para Google Gemini, usando el SDK oficial `google-genai`.

Traduce la interfaz común (system + messages) al formato de Gemini y devuelve
el texto y los tokens. Los reintentos NO se hacen acá: los hace LLMProvider.generate().
"""

from google import genai
from google.genai import errors, types

from src.providers.base import LLMProvider, Message, PermanentLLMError, TransientLLMError


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

        # El SDK espera el timeout en milisegundos.
        # No activamos los reintentos propios del SDK: de eso se encarga LLMProvider.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )

    def _call(self, system: str, messages: list[Message]) -> tuple[str, int, int]:
        # 1) Traducir los mensajes al formato de Gemini.
        #    Gemini llama "model" a lo que nosotros llamamos "assistant".
        contents = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message["content"]}]})

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        # 2) Llamar a la API y traducir los errores del SDK a nuestros dos tipos.
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except errors.APIError as error:
            # La API respondió con un código de error HTTP.
            if error.code == 429 or error.code >= 500:
                raise TransientLLMError(f"Gemini {error.code}: {error.message}") from error
            raise PermanentLLMError(f"Gemini {error.code}: {error.message}") from error
        except Exception as error:
            # La API no llegó a responder (timeout o problema de red).
            raise TransientLLMError(f"Gemini sin respuesta (timeout o red): {error}") from error

        # 3) Extraer el texto y los tokens.
        #    Los tokens de "razonamiento interno" (thoughts) se facturan como salida,
        #    por eso se suman a completion_tokens. El SDK devuelve None cuando un conteo es 0.
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count or 0
        completion_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        text = response.text or ""

        return text, prompt_tokens, completion_tokens
