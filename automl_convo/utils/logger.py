"""
This file defines the logger component with Rich logger. Makes the program look nice :D
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from typing import Optional
import os

console = Console()

class Logger:
    _file_handle: Optional[object] = None
    _log_file: Optional[str] = None

    def __init__(self, log_file: str = "logs/app.log"):
        # Initialize shared file only once
        if Logger._file_handle is None:
            Logger._log_file = log_file

            # Create parent directory if it doesn't exist
            directory = os.path.dirname(log_file)
            if directory:
                os.makedirs(directory, exist_ok=True)

            Logger._file_handle = open(log_file, "a", encoding="utf-8")

    def _write(self, message: str):
        if Logger._file_handle:
            Logger._file_handle.write(message + "\n")
            Logger._file_handle.flush()

    def box(self, title, content, style="cyan"):
        console.print(Panel(content, title=f"[bold]{title}[/bold]", style=style))
        self._write(f"[{title}] {content}")

    def md(self, content):
        console.print(Markdown(content))
        self._write(content)

    def info(self, content, style="white"):
        console.print(f"[{style}]{content}[/{style}]")
        self._write(content)

    def reasoning(self, content):
        console.print(Panel(content, title="[gray]LLM reasoning[/gray]", style="dim"))
        self._write(f"[LLM reasoning] {content}")

    def box_md(self, title: str, markdown_text: str, style: str = "cyan"):
        md = Markdown(markdown_text)
        console.print(Panel(md, title=f"[bold]{title}[/bold]", style=style))
        self._write(f"[{title}] {markdown_text}")
