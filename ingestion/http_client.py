import os
import warnings

_VERIFY_ENV = "HTTPX_SSL_VERIFY"


def get_ssl_verify() -> bool | str:
    """Resolve the SSL verification setting shared by all outbound httpx clients.

    Behind a corporate TLS-inspecting proxy, set HTTPX_SSL_VERIFY to the path of
    the proxy's root CA bundle (.pem) instead of disabling verification —
    that's the fix that actually restores trust rather than removing it.
    Setting it to "false" is a last resort: it disables protection against
    man-in-the-middle attacks on every outbound request, including the Tavily
    call that carries your API key.
    """
    value = os.getenv(_VERIFY_ENV)
    if not value or not value.strip():
        return True
    if value.strip().lower() in ("false", "0", "no"):
        warnings.warn(
            "HTTPX_SSL_VERIFY=false: outbound HTTPS requests are not verifying "
            "server certificates. Prefer setting HTTPX_SSL_VERIFY to your "
            "proxy's CA bundle path instead.",
            stacklevel=2,
        )
        return False
    return value.strip()  # path to a CA bundle file
