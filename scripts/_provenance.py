from urllib.parse import urlsplit, urlunsplit


def safe_source_url(value: str | None) -> tuple[str | None, bool]:
    if not value:
        return None, False
    parts = urlsplit(value)
    if parts.scheme.lower() == "file":
        return "local_file_import", bool(parts.query or parts.fragment)
    host = parts.hostname or ""
    try:
        if parts.port:
            host = f"{host}:{parts.port}"
    except ValueError:
        host = parts.hostname or ""
    sanitized = urlunsplit((parts.scheme, host, parts.path, "", ""))
    return sanitized, bool(parts.query or parts.fragment or parts.username or parts.password)
