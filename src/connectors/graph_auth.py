import os
from pathlib import Path

import msal
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_CACHE_PATH = PROJECT_ROOT / "data/auth/graph_token_cache.bin"

load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_graph_scopes() -> list[str]:
    """Read Microsoft Graph delegated scopes from environment config."""
    scopes_text = os.getenv("GRAPH_SCOPES", "User.Read Files.Read Notes.Read")
    return [
        scope.strip()
        for scope in scopes_text.split()
        if scope.strip()
    ]


def load_token_cache() -> msal.SerializableTokenCache:
    """Load the local MSAL token cache used for Graph delegated auth."""
    cache = msal.SerializableTokenCache()

    if TOKEN_CACHE_PATH.exists():
        cache.deserialize(TOKEN_CACHE_PATH.read_text(encoding="utf-8"))

    return cache


def save_token_cache(cache: msal.SerializableTokenCache) -> None:
    """Persist the MSAL token cache when it changes."""
    if not cache.has_state_changed:
        return

    TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE_PATH.write_text(cache.serialize(), encoding="utf-8")


def build_public_client(cache: msal.SerializableTokenCache) -> msal.PublicClientApplication:
    """Build the MSAL public client for delegated Microsoft Graph access."""
    client_id = os.getenv("GRAPH_CLIENT_ID")
    authority = os.getenv(
        "GRAPH_AUTHORITY",
        "https://login.microsoftonline.com/consumers",
    )

    if not client_id:
        raise RuntimeError("GRAPH_CLIENT_ID is required for Graph MSAL authentication.")

    return msal.PublicClientApplication(
        client_id=client_id,
        authority=authority,
        token_cache=cache,
    )


def get_graph_access_token() -> str:
    """Return a Graph access token from manual env fallback or MSAL cache."""
    manual_token = os.getenv("GRAPH_ACCESS_TOKEN")

    if manual_token:
        return manual_token

    cache = load_token_cache()
    app = build_public_client(cache)
    accounts = app.get_accounts()

    if not accounts:
        raise RuntimeError(
            "Graph sign-in is required. No cached Microsoft account was found."
        )

    result = app.acquire_token_silent(
        scopes=get_graph_scopes(),
        account=accounts[0],
    )

    save_token_cache(cache)

    if not result or "access_token" not in result:
        raise RuntimeError(
            "Graph sign-in is required. Cached token could not be refreshed."
        )

    return result["access_token"]


def run_device_login() -> dict:
    """Run a device-code sign-in and persist the resulting MSAL token cache."""
    cache = load_token_cache()
    app = build_public_client(cache)

    flow = app.initiate_device_flow(scopes=get_graph_scopes())

    if "user_code" not in flow:
        raise RuntimeError(f"Device login could not start: {flow}")

    print(flow["message"])

    result = app.acquire_token_by_device_flow(flow)

    save_token_cache(cache)

    if "access_token" not in result:
        raise RuntimeError(f"Device login failed: {result}")

    return {
        "status": "success",
        "account": result.get("id_token_claims", {}).get("preferred_username"),
        "scopes": get_graph_scopes(),
    }

