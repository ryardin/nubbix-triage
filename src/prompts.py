"""Carga los prompts versionados (prompts/*.yaml) y arma lo que recibe el proveedor.

Uso:
    from src.prompts import build_prompt
    system, messages = build_prompt("v2_few_shot", "Me cobraron dos veces")
    response = provider.generate(system, messages)

Los YAML tienen marcadores que se reemplazan acá:
    {context}    -> contenido de prompts/nubbix_context.yaml
    {categories} -> categorías con su descripción, tomadas de src/schemas.py
    {priority}   -> orden de desempate ("account > technical > ..."), tomado de src/schemas.py
    {actions}    -> acciones válidas agrupadas por categoría, tomadas de src/schemas.py
    {examples}   -> ejemplos resueltos de la versión (solo few-shot), como texto
    {ticket}     -> texto del cliente

Categorías, prioridad y acciones salen de schemas.py: una sola fuente de verdad.

Se usa .replace() y no .format(): el prompt contiene llaves reales (el JSON de
ejemplo del formato de salida) que .format() confundiría con marcadores.
"""

import yaml

from src.config import PROJECT_ROOT, ConfigError
from src.schemas import ACTIONS, CATEGORIES, TOOLS

PROMPTS_DIR = PROJECT_ROOT / "prompts"
CONTEXT_FILE = PROMPTS_DIR / "nubbix_context.yaml"


def load_yaml(path) -> dict:
    """Lee un archivo YAML y lo devuelve como diccionario."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_version(version: str) -> dict:
    """Lee el YAML de una versión de prompt, o avisa cuáles existen si no la encuentra."""
    path = PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in PROMPTS_DIR.glob("v*.yaml")))
        raise ConfigError(f"Versión de prompt desconocida: '{version}'. Opciones válidas: {available}.")
    return load_yaml(path)


def get_tool(version: str) -> dict:
    """Devuelve la tool que usa una versión de prompt (clave "tool" del YAML).

    Si el YAML no tiene la clave, usa "triage" (la tool del contrato de 4 campos).
    v3 (chain-of-thought) usa "triage_with_reasoning".
    """
    tool_name = load_version(version).get("tool", "triage")
    if tool_name not in TOOLS:
        raise ConfigError(f"Tool desconocida en '{version}': '{tool_name}'. Opciones válidas: {', '.join(TOOLS)}.")
    return TOOLS[tool_name]


def build_prompt(version: str, ticket: str) -> tuple[str, list[dict[str, str]]]:
    """Arma el system prompt y los mensajes para una versión de prompt y un ticket.

    Devuelve (system, messages), listo para provider.generate(system, messages, tool),
    con tool = get_tool(version).
    En las versiones few-shot, los ejemplos van como texto dentro del system prompt
    (marcador {examples}); messages lleva siempre un solo mensaje: el ticket real.
    """
    prompt = load_version(version)
    context = load_yaml(CONTEXT_FILE)["content"]

    # Categorías con su descripción, una por línea: "- account: acceso, contraseña, ..."
    categories = "\n".join(f"- {name}: {description}" for name, description in CATEGORIES.items())

    # Prioridad de desempate: el orden del diccionario CATEGORIES.
    priority = " > ".join(CATEGORIES)

    # Acciones válidas agrupadas, un grupo por línea:
    # "- common: escalate_to_human, request_more_info"
    actions = "\n".join(f"- {group}: {', '.join(names)}" for group, names in ACTIONS.items())

    system = prompt["system"].replace("{context}", context.strip())
    system = system.replace("{categories}", categories)
    system = system.replace("{priority}", priority)
    system = system.replace("{actions}", actions)

    # Ejemplos (solo en few-shot; en zero-shot la clave "examples" no existe y el
    # system no tiene el marcador). Cada uno queda así:
    #   Ejemplo 1
    #   Ticket: Olvidé mi contraseña...
    #   Respuesta: {"answer": ..., "confidence": 4, ...}
    examples = "\n\n".join(
        f"Ejemplo {number}\nTicket: {example['ticket'].strip()}\nRespuesta: {example['response'].strip()}"
        for number, example in enumerate(prompt.get("examples", []), start=1)
    )
    system = system.replace("{examples}", examples)

    # Un solo mensaje: el ticket real del cliente, entre los delimitadores.
    messages = [{"role": "user", "content": _fill_ticket(prompt, ticket)}]

    return system.strip(), messages


def _fill_ticket(prompt: dict, ticket: str) -> str:
    """Inserta el texto del cliente entre los delimitadores de user_template."""
    return prompt["user_template"].replace("{ticket}", ticket.strip()).strip()
