"""
config.py
Configurações compartilhadas para o sistema DEMATEL-Delphi
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Configurações de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-pro"
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Escala e prompts
SCALE_DESC = "0 = sem influência nenhuma, 1 = muito baixa ... 9 = muito alta"

PROMPT_TMPL = (
    "Você é um engenheiro Aeroespacial, especialista em propulsão de foguetes, com vasto conhecimento teórico, mas também prático. Você foi colocado em um projeto de construção de um motor foguete a propelente híbrido e está realizando um teste no modelo DEMATEL para categorizar a influencia entre fatores e sua causalidade.\n"
    
    "Considere o par de fatores abaixo no contexto do projeto de desenvolvimento de um motor foguete híbrido, cujo requisito principal de sucesso é o empuxo gerado pelo motor. Julgue conforme a pergunta a seguir:\n"
    
    "Em uma escala de 0 a 9 ({scale}), qual o nível em que **{src}** influencia **{tgt}**? \n\n"
    
    "Considere que muitos fatores podem não ter influência direta, então sinta-se à vontade para responder 0 se achar que não há influência significativa.\n\n"

    "\nEntenda\n -{src} como: {description_src};\n -{tgt} como: {description_tgt}\n\n"

    "Avalie diferentes cenários em que é possível se ter {src} e como variações (pequenas ou grandes) em {src} pode influenciar {tgt}. Atente-se à magnitude dessa influência, e não à sua direção (positiva ou negativa).\n"
    
    "Para a definição de influência, considere também, se {tgt} é um fator que pode ser afetado por {src} considerando a lógica do projeto e a sua participação. Antes de responder, pense na origem dos fatores (por exemplo, se são aspectos externos, se são aspectos de projetos, se referem apenas a atributos de propelente ou atributos estruturais) para definir se o aspecto target é realmente passível de ser alterado. É possível que {tgt} possa ter influência direta, mas que não possam ser alterados diretamente, no contexto de um projeto, por {src}.\n\n"
    
    "Entenda que o resultado deve ser interpretado no contexto do projeto e a influência é uma relação unidirecional. Associe com o contexto de causalidade de A em B.\n\n"

    "A resposta deve ser apenas o número."
)

# Detecção da versão do openai
try:
    from importlib.metadata import version as _pkg_version
    _OPENAI_V0 = _pkg_version("openai").startswith("0.")
except Exception:
    # Fallback heurístico
    import openai
    _OPENAI_V0 = hasattr(openai, "ChatCompletion") and not hasattr(openai, "OpenAI")

# Cliente OpenAI
if not _OPENAI_V0:
    import openai
    def _make_openai_client(**kwargs):
        return openai.OpenAI(**kwargs)
else:
    _make_openai_client = None