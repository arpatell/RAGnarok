"""Read an SSH-transported base64 key from stdin; update the existing RAG env."""
import base64
import os
from pathlib import Path
import sys
import tempfile


def update_env(path, updates):
    lines = path.read_text().splitlines()
    retained = [line for line in lines if line.partition("=")[0] not in updates]
    retained.extend(f"{key}={value}" for key, value in updates.items())
    metadata = path.stat()
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        staging = Path(handle.name)
        handle.write("\n".join(retained) + "\n")
    try:
        staging.chmod(metadata.st_mode & 0o777)
        if hasattr(os, "chown"):
            os.chown(staging, metadata.st_uid, metadata.st_gid)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


if __name__ == "__main__":
    key = base64.b64decode(sys.stdin.read().strip(), validate=True).decode().strip()
    if not key or any(c in key for c in "\r\n\x00"):
        raise SystemExit("Invalid Groq secret format")
    update_env(Path("/etc/ragnarok/rag.env"), {"GROQ_API_KEY": key, "GROQ_MODEL": "openai/gpt-oss-20b", "RAG_LLM_PROVIDER": "groq", "RAG_LLM_RERANK_ENABLED": "1"})
    update_env(Path("/etc/ragnarok/backend.env"), {"RAG_REQUEST_TIMEOUT_MS": "32000", "READ_NOW_TITLE_VARIANT_LIMIT": "2", "READ_NOW_TIMEOUT_MS": "5500", "READ_NOW_SERIES_RESOLVE_TIMEOUT_MS": "1800"})
    print("Groq configured in the existing RAG environment; key not displayed.")
