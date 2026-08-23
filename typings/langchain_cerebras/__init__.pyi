from typing import Any

from pydantic import SecretStr

class ChatCerebras:
    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0,
        api_key: SecretStr | None = None,
        timeout: float | None = None,
        max_retries: int = 0,
        **kwargs: Any,
    ) -> None: ...
    def invoke(self, messages: Any) -> Any: ...
