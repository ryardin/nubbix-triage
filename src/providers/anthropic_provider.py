"""Adaptador para Anthropic (Claude), usando el SDK oficial `anthropic`.

Los reintentos NO se hacen acá: los hace LLMProvider.generate().

Nota sobre temperature: los modelos actuales de Claude no aceptan este parámetro
(el SDK ni siquiera lo ofrece). Por eso este adaptador recibe `temperature` para
respetar la interfaz común, pero no lo envía a la API.
"""

import anthropic

from src.providers.base import LLMProvider, Message, PermanentLLMError, TransientLLMError


class AnthropicProvider(LLMProvider):
    name = "anthropic"

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

        # max_retries=0: el SDK de Anthropic reintenta 2 veces por defecto, en silencio.
        # Lo apagamos para que los reintentos los maneje (y los cuente) LLMProvider.
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)

    def _call(self, system: str, messages: list[Message]) -> tuple[str, int, int]:
        # 1) Traducir los mensajes: Anthropic recibe el system prompt aparte (parámetro `system`)
        #    y usa los mismos roles "user" y "assistant" que nuestro formato.
        anthropic_messages = []
        for message in messages:
            anthropic_messages.append({"role": message["role"], "content": message["content"]})

        # 2) Llamar a la API y traducir los errores del SDK a nuestros dos tipos.
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system,
                messages=anthropic_messages,
                max_tokens=self.max_tokens,
            )
        except anthropic.APIStatusError as error:
            # La API respondió con un código de error HTTP.
            if error.status_code == 429 or error.status_code >= 500:
                raise TransientLLMError(f"Anthropic {error.status_code}: {error.message}") from error
            raise PermanentLLMError(f"Anthropic {error.status_code}: {error.message}") from error
        except anthropic.APIConnectionError as error:
            # La API no llegó a responder (timeout o problema de red).
            raise TransientLLMError(f"Anthropic sin respuesta (timeout o red): {error}") from error

        # 3) Extraer el texto y los tokens.
        #    La respuesta es una lista de "bloques"; puede haber bloques de razonamiento
        #    ("thinking") además del texto. Nos quedamos solo con los bloques de texto.
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        #    output_tokens ya incluye los tokens de razonamiento, si los hubo.
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        return text, prompt_tokens, completion_tokens
