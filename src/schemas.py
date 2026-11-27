"""Contrato de salida del servicio: {answer, confidence, category, actions}.

El modelo devuelve la respuesta llamando a la tool TRIAGE_TOOL (definida al final),
cuyo schema sale de TriageResponse.
Toda respuesta del LLM se valida contra TriageResponse antes de devolverse.
Si no cumple el contrato, Pydantic lanza ValidationError y el pipeline reintenta
pasándole al modelo el error. Las decisiones de cada límite están en el README.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Categorías fijas por consigna: ÚNICA fuente de verdad. De este diccionario salen:
# - el tipo Category que valida Pydantic (abajo),
# - la lista con descripciones que recibe el prompt (marcador {categories}),
# - la regla de desempate (marcador {priority}): el ORDEN del diccionario es la prioridad
#   para tickets mixtos, según qué bloquea más al cliente.
CATEGORIES = {
    "account": "acceso, contraseña, usuarios, permisos y datos de la cuenta.",
    "technical": "errores, fallas, lentitud o algo de Nubbix que no funciona.",
    "billing": "pagos, cobros, facturas de Nubbix, planes y reembolsos.",
    "other": "el ticket no entra en ninguna de las categorías anteriores.",
}

# Literal.__getitem__(...) es lo mismo que escribir Literal["account", "technical", ...],
# pero construido desde CATEGORIES: si se agrega una categoría, el contrato se actualiza solo.
Category = Literal.__getitem__(tuple(CATEGORIES))

# Lista cerrada de acciones sugeridas para el agente humano, agrupadas por categoría:
# ÚNICA fuente de verdad. De este diccionario salen:
# - el tipo Action que valida Pydantic (abajo),
# - la lista agrupada que recibe el prompt (marcador {actions}).
# La agrupación orienta al modelo, pero la validación es flexible: un ticket puede
# llevar acciones de cualquier grupo (por ejemplo, un ticket mixto).
ACTIONS = {
    "common": ["escalate_to_human", "request_more_info"],
    "billing": ["review_payment_status", "check_duplicate_charge", "resend_invoice", "update_billing_details"],
    "technical": ["check_service_status", "send_troubleshooting_guide", "escalate_to_engineering"],
    "account": ["send_password_reset", "verify_identity", "update_account_details", "manage_user_permissions"],
    "other": ["forward_to_sales", "send_help_center_link"],
}

# Todas las acciones de todos los grupos, en una sola tupla, convertidas en Literal
# (igual que Category): si se agrega una acción a ACTIONS, el contrato se actualiza solo.
ALL_ACTIONS = tuple(action for group in ACTIONS.values() for action in group)
Action = Literal.__getitem__(ALL_ACTIONS)


class TriageResponse(BaseModel):
    """Respuesta estructurada del triage de un ticket de soporte."""

    # extra="forbid": un campo que no sea uno de estos 4 invalida la respuesta.
    # str_strip_whitespace=True: recorta espacios antes de validar (un answer de
    # puros espacios queda vacío y falla).
    # Los tipos son permisivos (por defecto): "3" o 3.0 se aceptan como 3.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Respuesta propuesta al cliente. El prompt pide 1 a 3 oraciones; el máximo
    # de 500 caracteres es el límite duro por si el modelo no lo respeta.
    answer: str = Field(min_length=5, max_length=500)

    # Qué tan segura es la CLASIFICACIÓN (no la respuesta). Escala par, sin punto medio:
    # 4 = categoría explícita (o seguro que es "other"), 3 = deducida,
    # 2 = ticket vago (falta información), 1 = ticket mixto (hubo desempate).
    confidence: int = Field(ge=1, le=4)

    category: Category

    # Entre 1 y 3 acciones: una lista vacía no le sirve al agente; más de 3 es ruido.
    actions: list[Action] = Field(min_length=1, max_length=3)


class TriageWithReasoning(TriageResponse):
    """TriageResponse + el razonamiento del modelo (solo para el prompt v3, chain-of-thought).

    Es un formato INTERNO: el pipeline valida con esta clase, separa `reasoning`
    (para métricas y evals) y devuelve un TriageResponse. El contrato público
    sigue siendo de 4 campos.
    """

    # Razonamiento breve paso a paso. El máximo acota los tokens de salida
    # (Groq tiene max_tokens=512 y los 4 campos del contrato también ocupan lugar).
    reasoning: str = Field(min_length=10, max_length=600)


# Tool que el modelo está OBLIGADO a llamar (tool calling forzado). Sus parámetros
# son el JSON Schema que Pydantic genera desde TriageResponse: el contrato se escribe
# una sola vez, y si cambia (por ejemplo, una categoría nueva), la tool cambia sola.
# "model" no se envía al proveedor: es la clase con la que el pipeline valida la respuesta.
TRIAGE_TOOL = {
    "name": "triage_ticket",
    "description": "Registra el triage de un ticket de soporte de Nubbix.",
    "schema": TriageResponse.model_json_schema(),
    "model": TriageResponse,
}


def _reasoning_first(schema: dict) -> dict:
    """Pone `reasoning` como PRIMER campo del schema.

    El modelo genera los campos en orden: si el razonamiento va primero, razona
    ANTES de decidir la categoría. Si fuera al final (como queda por herencia),
    razonaría después de haber decidido, y no serviría.
    """
    properties = schema["properties"]
    schema["properties"] = {"reasoning": properties.pop("reasoning"), **properties}
    schema["required"] = ["reasoning"] + [name for name in schema["required"] if name != "reasoning"]
    return schema


# Tool del prompt v3 (chain-of-thought): mismos campos + "reasoning" primero.
TRIAGE_WITH_REASONING_TOOL = {
    "name": "triage_ticket",
    "description": "Registra el triage de un ticket de soporte de Nubbix, con el razonamiento previo.",
    "schema": _reasoning_first(TriageWithReasoning.model_json_schema()),
    "model": TriageWithReasoning,
}

# Tools disponibles para los prompts: cada YAML elige una con la clave "tool"
# (si no la tiene, usa "triage").
TOOLS = {
    "triage": TRIAGE_TOOL,
    "triage_with_reasoning": TRIAGE_WITH_REASONING_TOOL,
}
