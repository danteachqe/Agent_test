"""
ToolAgent demo — fixed for OpenAI SDK v1.x

- Uses `from openai import OpenAI` and `client.chat.completions.create(...)`
- Replaces retired model with a current one (gpt-4o-mini by default)
- Keeps rule-based fallback if no OPENAI_API_KEY
"""

from __future__ import annotations

import ast
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, Tuple

try:
    # OpenAI SDK v1.x
    from openai import OpenAI  # type: ignore
    _openai_available = True
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore
    _openai_available = False

# Configure logging
logger = logging.getLogger("ToolAgent")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)


class ToolAgent:
    """A tiny agent that chooses between three hard-coded tools using OpenAI or a rule-based fallback."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.tools: Dict[str, Tuple[Callable[..., Any], str]] = {
            "calculator": (self._tool_calculator, "Evaluate a math expression."),
            "read_file": (self._tool_read_file, "Read a local file (path, max_chars)."),
            "timestamp": (self._tool_timestamp, "Return current time and date."),
        }
        self._client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if _openai_available and api_key:
            try:
                self._client = OpenAI()  # uses env var OPENAI_API_KEY
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client: %s", e)
                self._client = None

    # ---------- Tools ----------
    def _tool_calculator(self, expr: str) -> str:
        """Safely evaluate a math expression using AST parsing."""
        try:
            node = ast.parse(expr, mode="eval")
            if not self._is_safe_expr(node):
                return "Error: Unsafe or unsupported expression."
            result = eval(compile(node, filename="<ast>", mode="eval"), {}, {})
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"

    def _is_safe_expr(self, node: ast.AST) -> bool:
        """Recursively check AST nodes for safety: only allow numbers and operators."""
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Num,       # Py<3.8
            ast.Constant,  # Py>=3.8
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
            ast.USub, ast.UAdd, ast.Load, ast.FloorDiv,
            ast.Tuple, ast.List,
        )
        if isinstance(node, allowed):
            return all(self._is_safe_expr(child) for child in ast.iter_child_nodes(node))
        return False

    def _tool_read_file(self, path: str, max_chars: int = 500) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read(max_chars)
            return data
        except Exception as e:
            return f"Error reading file: {e}"

    def _tool_timestamp(self) -> str:
        now = datetime.utcnow()
        return now.isoformat() + "Z | " + now.strftime("%Y-%m-%d %H:%M:%S UTC")

    # ---------- Chooser ----------
    def choose_tool_with_openai(self, user_task: str) -> str:
        """Use OpenAI Chat Completions to choose a tool key."""
        if self._client is None:
            raise RuntimeError("OpenAI client unavailable (package not installed or no API key)")

        tools_list = "\n".join(f"- {k}: {v[1]}" for k, v in self.tools.items())
        prompt = (
            "You are an assistant that chooses which tool to use.\n"
            "Available tools:\n"
            f"{tools_list}\n\n"
            "Return ONLY the key name of the best tool for the user task. "
            "Valid keys: "
            f"{', '.join(self.tools.keys())}\n\n"
            f"User task: {user_task}\n"
        )

        logger.debug("Calling OpenAI to choose tool (model=%s)", self.model)

        # OpenAI SDK v1.x
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Choose the best tool key."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        chosen = text.split()[0].strip().lower()

        if chosen not in self.tools:
            logger.warning("OpenAI returned '%s' which is not a valid tool. Falling back.", chosen)
            raise ValueError(f"invalid tool: {chosen}")
        return chosen

    def choose_tool_rule_based(self, user_task: str) -> str:
        lowered = user_task.lower()
        if any(w in lowered for w in ("calculate", "compute", "evaluate", "math", "+", "-", "*", "/", "**")):
            return "calculator"
        if any(w in lowered for w in ("read", "open", "file", "load", "path", "c:\\", "./", "\\\\")):
            return "read_file"
        return "timestamp"

    # ---------- Public API ----------
    def handle(self, user_task: str) -> Dict[str, Any]:
        """Decide on a tool, run it, and return a structured log of the
