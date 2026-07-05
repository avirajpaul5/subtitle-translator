from __future__ import annotations

import os


SARVAM_API_KEY_ENV = "SARVAM_API_KEY"
_KEYRING_SERVICE = "indicsub"
_LEGACY_KEYRING_SERVICES = ("subtitle" + "-" + "translator",)
_SARVAM_KEYRING_ACCOUNT = "sarvam-api-key"


class CredentialStorageError(RuntimeError):
    pass


def get_sarvam_api_key(
    explicit_api_key: str | None = None,
    *,
    use_keyring: bool = True,
) -> str | None:
    """Return a Sarvam API key from explicit input, env, then OS keychain.

    Keys are never read from project files. The optional keychain path uses the
    user's OS-backed keyring when the `keyring` package is available.
    """
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    env_key = os.environ.get(SARVAM_API_KEY_ENV, "").strip()
    if env_key:
        return env_key

    if not use_keyring:
        return None

    try:
        import keyring
    except Exception:
        return None

    for service in (_KEYRING_SERVICE, *_LEGACY_KEYRING_SERVICES):
        try:
            stored = keyring.get_password(service, _SARVAM_KEYRING_ACCOUNT)
        except Exception:
            continue

        if stored and stored.strip():
            return stored.strip()
    return None


def save_sarvam_api_key(api_key: str) -> None:
    api_key = api_key.strip()
    if not api_key:
        raise CredentialStorageError("Cannot save an empty Sarvam API key.")

    try:
        import keyring
    except Exception as exc:
        raise CredentialStorageError(
            "Install the optional 'keyring' package to save keys in the OS keychain."
        ) from exc

    try:
        keyring.set_password(_KEYRING_SERVICE, _SARVAM_KEYRING_ACCOUNT, api_key)
    except Exception as exc:
        raise CredentialStorageError(f"Could not save key to OS keychain: {exc}") from exc


def has_stored_sarvam_api_key() -> bool:
    return bool(get_sarvam_api_key(use_keyring=True))
