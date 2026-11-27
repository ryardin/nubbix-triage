"""Adaptador para Groq, usando el SDK oficial `groq`.

Groq usa el mismo formato de mensajes que OpenAI, así que la traducción es mínima.
Los reintentos NO se hacen acá: los hace LLMProvider.generate().
"""

import groq

from src.providers.base import LLMProvider, Message, PermanentLLMError, Tool, TransientLLMError


class GroqProvider(LLMProvider):
    name = "groq"

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

        # max_retries=0: el SDK de Groq reintenta 2 veces por defecto, en silencio.
        # Lo apagamos para que los reintentos los maneje (y los cuente) LLMProvider.
        self.client = groq.Groq(api_key=api_key, timeout=timeout_s, max_retries=0)

    def _call(self, system: str, messages: list[Message], tool: Tool | None) -> tuple[str, int, int]:
        # 1) Traducir los mensajes: en Groq el system prompt es un mensaje más, al principio.
        #    Los roles "user" y "assistant" se llaman igual que en nuestro formato.
        groq_messages = [{"role": "system", "content": system}]
        for message in messages:
            groq_messages.append({"role": message["role"], "content": message["content"]})

        tool_params = {}
        if tool:
            tool_params = {
                # La tool en formato estilo OpenAI: el schema va en "parameters".
                "tools": [{"type": "function", "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["schema"],
                }}],
                # Obliga al modelo a llamar ESTA tool.
                "tool_choice": {"type": "function", "function": {"name": tool["name"]}},
            }

        # 2) Llamar a la API y traducir los errores del SDK a nuestros dos tipos.
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **tool_params,
            )
        except groq.APIStatusError as error:
            # La API respondió con un código de error HTTP.
            if error.status_code == 429 or error.status_code >= 500:
                raise TransientLLMError(f"Groq {error.status_code}: {error.message}") from error
            raise PermanentLLMError(f"Groq {error.status_code}: {error.message}") from error
        except groq.APIConnectionError as error:
            # La API no llegó a responder (timeout o problema de red).
            raise TransientLLMError(f"Groq sin respuesta (timeout o red): {error}") from error

        # 3) Extraer el texto (o los argumentos de la tool) y los tokens.
        #    En Groq, completion_tokens YA incluye los tokens de razonamiento interno.
        message = response.choices[0].message
        if tool:
            # Groq entrega los argumentos como string JSON: ya están en el formato que queremos.
            calls = message.tool_calls or []
            text = calls[0].function.arguments if calls else ""
        else:
            text = message.content or ""
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        return text, prompt_tokens, completion_tokens
