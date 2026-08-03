"""Auth endpoints (#19): local login/logout and OIDC login/callback.

Deliberately mounted in main.py WITHOUT the bearer-key/session guard — login
has to be reachable by an unauthenticated browser. `/auth/oidc/*` 404s
outright when `settings.oidc_issuer` is unset, so an unconfigured deployment
exposes nothing new.

OIDC uses Authlib's `AsyncOAuth2Client` for the code exchange and `joserfc`
(Authlib's own successor to `authlib.jose`) to verify the id_token's
signature against the issuer's JWKS — never `verify_signature: False`. The
whole exchange-and-verify step is one function, `_exchange_code_for_claims`,
so it can be monkeypatched at the network boundary in tests.

Group claims only ever SEED membership (#2's binding resolution): a login
with a `groups` claim replaces this user's *idp-sourced* group rows —
locally-created group membership is untouched — and later requests read the
stored `user_groups` rows, never the token.
"""

import secrets
from datetime import UTC, datetime

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from joserfc import jwt
from joserfc.jwk import KeySet
from joserfc.jwt import JWTClaimsRegistry
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.auth import SESSION_COOKIE_NAME, create_session, hash_key, verify_password
from prorag.db import get_session
from prorag.models import Group, Session, User, UserGroup
from prorag.settings import settings

router = APIRouter()

OIDC_STATE_COOKIE = "prorag_oidc_state"


def _set_session_cookie(response: Response, raw_token: str) -> None:
    """When called: by login and oidc_callback after minting a session token.
    What: sets the `prorag_session` cookie (HttpOnly, SameSite=lax, secure
    per settings, TTL = session_ttl_days) on the response."""
    response.set_cookie(
        SESSION_COOKIE_NAME,
        raw_token,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        max_age=settings.session_ttl_days * 86400,
        path="/",
    )


def _redirect_uri(request: Request) -> str:
    """When called: by oidc_login and oidc_callback to build the issuer-facing
    redirect_uri. What: derives it from the incoming request's base URL (so a
    proxy's X-Forwarded-Proto is honored) plus settings.oidc_redirect_path."""
    # Deliberately built from the request rather than a PUBLIC_BASE_URL
    # setting — #5's research flags this as the #1 OIDC deployment bug behind
    # a proxy that doesn't forward X-Forwarded-Proto. ponytail: fine for the
    # single-box deployment this repo targets today; add PUBLIC_BASE_URL if a
    # reverse-proxied deployment needs it.
    return str(request.base_url).rstrip("/") + settings.oidc_redirect_path


async def _discovery(issuer: str) -> dict:
    """When called: at the start of oidc_login and the code exchange. What:
    fetches the issuer's OpenID Connect discovery document from the
    well-known endpoint. Returns: the parsed JSON metadata dict."""
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def _verify_id_token(id_token: str, meta: dict, nonce: str | None) -> dict:
    """Verifies the id_token's signature against the issuer's live JWKS
    (Authlib/joserfc), plus iss/aud/exp and the state-bound nonce."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        jwks_resp = await client.get(meta["jwks_uri"])
        jwks_resp.raise_for_status()
    key_set = KeySet.import_key_set(jwks_resp.json())
    token = jwt.decode(id_token, key_set)
    registry = JWTClaimsRegistry(
        iss={"essential": True, "value": settings.oidc_issuer},
        aud={"essential": True, "value": settings.oidc_client_id},
        exp={"essential": True},
    )
    registry.validate(token.claims)
    if nonce and token.claims.get("nonce") != nonce:
        raise HTTPException(400, "oidc nonce mismatch")
    return token.claims


async def _exchange_code_for_claims(code: str, redirect_uri: str, nonce: str | None) -> dict:
    """The network+verification boundary: discovery, code exchange, JWKS
    fetch, signature check. Tests monkeypatch this whole function."""
    meta = await _discovery(settings.oidc_issuer)
    client = AsyncOAuth2Client(settings.oidc_client_id, settings.oidc_client_secret)
    token = await client.fetch_token(meta["token_endpoint"], code=code, redirect_uri=redirect_uri)
    id_token = token.get("id_token")
    if not id_token:
        raise HTTPException(400, "oidc token response missing id_token")
    return await _verify_id_token(id_token, meta, nonce)


async def _seed_idp_groups(session: AsyncSession, user_id, group_claim: list) -> None:
    """Replaces this user's idp-sourced group rows with the token's current
    `groups` claim. Local-sourced memberships are a disjoint set (different
    group rows) and are never touched here — #2's rule that claims seed
    membership, they don't get trusted per-request afterward."""
    idp_group_ids = select(Group.id).where(Group.source == "idp")
    await session.execute(delete(UserGroup).where(UserGroup.user_id == user_id, UserGroup.group_id.in_(idp_group_ids)))
    for raw_external_id in group_claim:
        external_id = str(raw_external_id)
        group = (
            await session.execute(select(Group).where(Group.source == "idp", Group.external_id == external_id))
        ).scalar_one_or_none()
        if group is None:
            group = Group(name=f"idp:{external_id}", source="idp", external_id=external_id)
            session.add(group)
            await session.flush()
        session.add(UserGroup(user_id=user_id, group_id=group.id))


class LoginRequest(BaseModel):
    """When called: parsed by FastAPI from the POST /auth/login JSON body.
    What: carries the local login credentials."""
    email: str
    password: str


@router.post("/auth/login")
async def login(req: LoginRequest, session: AsyncSession = Depends(get_session)):
    """When called: POST /auth/login with email+password. What: verifies the
    password against the stored hash (401 for unknown/disabled user or wrong
    password), then mints a session and sets the cookie. Returns: JSONResponse
    with the user's email/display_name plus the session cookie."""
    user = (await session.execute(select(User).where(User.email == req.email))).scalar_one_or_none()
    if (
        user is None
        or user.password_hash is None
        or user.disabled_at is not None
        or not verify_password(user.password_hash, req.password)
    ):
        raise HTTPException(401, "invalid email or password")

    raw_token = await create_session(user.id, session)
    resp = JSONResponse({"email": user.email, "display_name": user.display_name})
    _set_session_cookie(resp, raw_token)
    return resp


@router.post("/auth/logout")
async def logout(request: Request, session: AsyncSession = Depends(get_session)):
    """When called: POST /auth/logout. What: revokes the session row behind
    the cookie (if any) and clears the cookie. Returns: JSONResponse {"ok":
    True}."""
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if raw:
        row = (
            await session.execute(select(Session).where(Session.token_hash == hash_key(raw)))
        ).scalar_one_or_none()
        if row is not None and row.revoked_at is None:
            row.revoked_at = datetime.now(UTC)
            await session.commit()
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return resp


@router.get("/auth/oidc/login")
async def oidc_login(request: Request):
    """When called: GET /auth/oidc/login. What: starts OIDC — fetches
    discovery, builds the authorization URL with a fresh nonce, and redirects
    the browser to the issuer with a state+nonce cookie. Returns:
    RedirectResponse to the issuer; 404 when OIDC is unconfigured."""
    if not settings.oidc_issuer:
        raise HTTPException(404)

    meta = await _discovery(settings.oidc_issuer)
    nonce = secrets.token_urlsafe(16)
    client = AsyncOAuth2Client(
        settings.oidc_client_id,
        settings.oidc_client_secret,
        scope="openid email profile",
        redirect_uri=_redirect_uri(request),
    )
    uri, state = client.create_authorization_url(meta["authorization_endpoint"], nonce=nonce)

    resp = RedirectResponse(uri)
    resp.set_cookie(
        OIDC_STATE_COOKIE,
        f"{state}:{nonce}",
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        max_age=600,
        path="/",
    )
    return resp


@router.get(settings.oidc_redirect_path)
async def oidc_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """When called: GET {oidc_redirect_path} — the issuer redirects the
    browser here after the user consents. What: verifies state+nonce against
    the cookie, exchanges the code for verified claims, upserts the user,
    seeds idp group membership from the token's `groups` claim, and sets a
    session cookie. Returns: RedirectResponse to "/"."""
    if not settings.oidc_issuer:
        raise HTTPException(404)
    if not code or not state:
        raise HTTPException(400, "missing code or state")

    cookie_val = request.cookies.get(OIDC_STATE_COOKIE)
    if not cookie_val or ":" not in cookie_val:
        raise HTTPException(400, "missing oidc state cookie")
    saved_state, nonce = cookie_val.split(":", 1)
    if not secrets.compare_digest(saved_state, state):
        raise HTTPException(400, "oidc state mismatch")

    claims = await _exchange_code_for_claims(code, _redirect_uri(request), nonce)

    email = claims.get("email")
    sub = claims.get("sub")
    if not email or not sub:
        raise HTTPException(400, "id token missing required email/sub claim")

    user = (await session.execute(select(User).where(User.external_subject == sub))).scalar_one_or_none()
    if user is None:
        user = User(external_subject=sub, email=email, display_name=claims.get("name"))
        session.add(user)
        await session.flush()
    else:
        user.email = email
        if claims.get("name"):
            user.display_name = claims.get("name")

    groups_claim = claims.get("groups")
    if groups_claim is not None:
        await _seed_idp_groups(session, user.id, groups_claim)

    raw_token = await create_session(user.id, session)  # commits the user/group changes above too

    resp = RedirectResponse(url="/", status_code=302)
    _set_session_cookie(resp, raw_token)
    resp.delete_cookie(OIDC_STATE_COOKIE, path="/")
    return resp
