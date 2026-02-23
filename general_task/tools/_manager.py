"""Tool 管理器：自動掃描 tools/ 目錄載入所有工具"""

from __future__ import annotations

import functools
import importlib
import os
from pathlib import Path
from typing import List, Optional

from google.adk.tools import FunctionTool


class ToolManager:
    """Tool 管理器。單例類別。"""

    _instance: Optional[ToolManager] = None
    _initialized: bool

    def __new__(cls) -> ToolManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.tools: List[FunctionTool] = []
        self.tool_prompts: List[str] = []
        self._load_tools_from_dir()
        self._load_tool_prompts_from_dir()

    def _load_tools_from_dir(self) -> None:
        """從 tools dir 載入所有工具"""
        dir_path = Path(__file__).parent
        for file in sorted(os.listdir(dir_path)):
            if not file.endswith(".py"):
                continue
            if file.startswith("__") or file.startswith("_"):
                continue
            module_name = file.replace(".py", "")
            module = importlib.import_module(f".{module_name}", package=__package__)
            tool_name = f"{module_name}_tool"
            if hasattr(module, tool_name):
                tool = getattr(module, tool_name)
                print(f"Loaded tool: {tool_name}")
                self.tools.append(tool)

    def _load_tool_prompts_from_dir(self) -> None:
        """從 tools/prompts dir 載入所有 tool 的 prompt"""
        dir_path = Path(__file__).parent / "prompts"
        if not dir_path.exists():
            return
        for file in sorted(os.listdir(dir_path)):
            if not file.endswith(".md"):
                continue
            if file.startswith("__") or file.startswith("_"):
                continue
            tool_prompt = (dir_path / file).read_text(encoding="utf-8")
            self.tool_prompts.append(tool_prompt)

    @staticmethod
    def _generate_alias_suffixes(max_depth: int = 3) -> list[str]:
        """Generate all permutations of noise tokens up to max_depth.

        gpt-oss models hallucinate suffixes like <|channel|>commentary, json,
        and combinations thereof. Rather than hardcoding each pattern, we
        generate all products of the three tokens up to max_depth so any
        future combination is automatically covered.
        """
        import itertools

        noise_tokens = ["json", "commentary", "<|channel|>"]
        suffixes: set[str] = set()
        for depth in range(1, max_depth + 1):
            for combo in itertools.product(noise_tokens, repeat=depth):
                suffixes.add("".join(combo))
        return list(suffixes)

    def get_all_tools(self, include_aliases: bool = False) -> List[FunctionTool]:
        tools = list(self.tools)
        if include_aliases:
            for tool in self.tools:
                func = tool.func
                for suffix in self._generate_alias_suffixes():
                    alias_name = f"{func.__name__}{suffix}"

                    @functools.wraps(func)
                    def wrapper(*args: object, _f: object = func, **kwargs: object) -> object:
                        return _f(*args, **kwargs)  # type: ignore[operator]

                    wrapper.__name__ = alias_name
                    tools.append(FunctionTool(wrapper))
        return tools

    def get_all_tool_prompts(self) -> List[str]:
        return self.tool_prompts


tool_manager = ToolManager()
