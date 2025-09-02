from __future__ import annotations
import os, logging, secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  SECRET_KEY ***must*** be provided in the environment in production.
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    log.critical(
        "ENV variable SECRET_KEY is missing -- generating a temporary key. "
        "ALL issued JWTs will be invalid after a pod restart! "
        "Set it in Railway → Variables to disable this warning."
    )
    SECRET_KEY = secrets.token_urlsafe(32)   # fallback only for dev

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token")

class TokenData(BaseModel):
    username: Optional[str] = None

class LoginPayload(BaseModel):
    username: str
    password: str

async def get_credentials(request: Request) -> LoginPayload:
    """
    Accept either JSON **or** classic form‑encoded credentials.

    Order of precedence:
    1. If the request media‑type is JSON → parse it with Pydantic.
    2. Else parse as form-encoded data.
    """
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        # JSON branch
        try:
            body = await request.json()
            return LoginPayload(**body)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid JSON credentials: {e}",
            )
    else:
        # Form-encoded branch
        try:
            form_data = await request.form()
            username = form_data.get("username")
            password = form_data.get("password")

            if not username or not password:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="username and password are required"
                )

            return LoginPayload(username=username, password=password)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid form credentials: {e}",
            )

def verify_password(raw: str, hashed: str) -> bool:
    return pwd_ctx.verify(raw, hashed)

def get_password_hash(pw: str) -> str:
    return pwd_ctx.hash(pw)

def create_access_token(subject: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": subject, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return username
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED) from exc
