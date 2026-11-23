"""Contrato de salida del servicio: {answer, confidence, category, actions}.

Toda respuesta del LLM se valida contra TriageResponse antes de devolverse.
Si no cumple el contrato, Pydantic lanza ValidationError y el pipeline reintenta
pasándole al modelo el error. Las decisiones de cada límite están en el README.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Categorías fijas por consigna, escritas en orden de prioridad (qué bloquea más al
# cliente): account > technical > billing > other. El orden acá es solo visual: el
# desempate de tickets mixtos lo hace el modelo, siguiendo la regla del prompt.
Category = Literal["account", "technical", "billing", "other"]

# Lista cerrada de acciones sugeridas para el agente humano. Es flexible: un
# ticket puede llevar acciones de otra categoría (por ejemplo, un ticket mixto).
Action = Literal[
    # Comunes a todas las categorías
    "escalate_to_human",
    "request_more_info",
    # billing
    "review_payment_status",
    "check_duplicate_charge",
    "resend_invoice",
    "update_billing_details",
    # technical
    "check_service_status",
    "send_troubleshooting_guide",
    "escalate_to_engineering",
    # account
    "send_password_reset",
    "verify_identity",
    "update_account_details",
    "manage_user_permissions",
    # other
    "forward_to_sales",
    "send_help_center_link",
]


class TriageResponse(BaseModel):
    """Respuesta estructurada del triage de un ticket de soporte."""

    # extra="forbid": un campo que no sea uno de estos 4 invalida la respuesta.
    # str_strip_whitespace=True: recorta espacios antes de validar (un answer de
    # puros espacios queda vacío y falla).
    # Los tipos son permisivos (por defecto): "3" o 3.0 se aceptan como 3.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Respuesta propuesta al cliente. El prompt pide 2 a 3 oraciones; el máximo
    # de 500 caracteres es el límite duro por si el modelo no lo respeta.
    answer: str = Field(min_length=5, max_length=500)

    # Qué tan segura es la CLASIFICACIÓN (no la respuesta). Escala par, sin punto medio:
    # 4 = categoría explícita (o seguro que es "other"), 3 = deducida,
    # 2 = ticket vago (falta información), 1 = ticket mixto (hubo desempate).
    confidence: int = Field(ge=1, le=4)

    category: Category

    # Entre 1 y 3 acciones: una lista vacía no le sirve al agente; más de 3 es ruido.
    actions: list[Action] = Field(min_length=1, max_length=3)
