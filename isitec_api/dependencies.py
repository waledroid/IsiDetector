import os
import secrets
from fastapi import Header, HTTPException

# ── Dev-mode authentication ──────────────────────────────────────────────────
DEV_PASSWORD = os.environ.get('DEV_PASSWORD', 'Isitec69+')
_dev_tokens: set[str] = set()


def create_dev_token() -> str:
    token = secrets.token_hex(16)
    _dev_tokens.add(token)
    return token


def discard_dev_token(token: str):
    _dev_tokens.discard(token)


def check_dev_token(token: str) -> bool:
    return token in _dev_tokens


def require_dev(x_dev_token: str = Header("", alias="X-Dev-Token")):
    """FastAPI dependency — raises 403 if dev token is missing or invalid."""
    if x_dev_token not in _dev_tokens:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return x_dev_token
