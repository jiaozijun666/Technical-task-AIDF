import os
try:
    from src.secrets_local import HF_TOKEN as HF_TOKEN_LOCAL  # type: ignore
except Exception:
    HF_TOKEN_LOCAL = None

try:
    from src.secrets_local import OPENAI_API_KEY as OPENAI_API_KEY_LOCAL, OPENAI_BASE_URL as OPENAI_BASE_URL_LOCAL  # type: ignore
except Exception:
    OPENAI_API_KEY_LOCAL = None
    OPENAI_BASE_URL_LOCAL = None