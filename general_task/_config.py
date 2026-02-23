"""共享配置單例模組"""

from __future__ import annotations

from typing import Optional


class _AgentConfig:
    """用於存儲 agent 運行時的配置，供各個模組使用。單例類別。"""

    _instance: Optional[_AgentConfig] = None
    _initialized: bool

    def __new__(cls) -> _AgentConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._api_key: Optional[str] = None
        self._api_base: Optional[str] = None
        self._model_name: Optional[str] = None
        self._output_dir: str = "./tmp/general_task_output"

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        self._api_key = value

    @property
    def api_base(self) -> Optional[str]:
        return self._api_base

    @api_base.setter
    def api_base(self, value: Optional[str]) -> None:
        self._api_base = value

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @model_name.setter
    def model_name(self, value: Optional[str]) -> None:
        self._model_name = value

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: str) -> None:
        self._output_dir = value


agent_config = _AgentConfig()
