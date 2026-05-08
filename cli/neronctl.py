"""neronctl — CLI for the Neron LLM service.

Usage:
    python -m cli.neronctl "Qui es-tu ?" --task default --mode single
    python -m cli.neronctl "Compare ces deux approches" --mode parallel
    python -m cli.neronctl "Reponds vite" --mode race
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.syntax import Syntax

from core.manager import LLMManager
from core.types import LLMRequest

app = typer.Typer(name="neronctl", help="Interface CLI pour le service LLM de Neron")
console = Console()
manager = LLMManager()


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message a envoyer au LLM"),
    task: str = typer.Option("default", "--task", "-t", help="Tache de routage (code, chat, default...)"),
    mode: str = typer.Option(None, "--mode", "-m", help="single | parallel | race (auto if unset)"),
    provider: str = typer.Option(None, "--provider", "-p", help="Forcer un provider (ollama, claude)"),
    model: str = typer.Option(None, "--model", help="Forcer un modele specifique"),
    pretty: bool = typer.Option(False, "--pretty", help="Afficher la reponse en JSON formate"),
):
    """Envoie un message au LLM et affiche la reponse."""
    req = LLMRequest(message=message, task=task, mode=mode, provider=provider, model=model)

    with console.status(f"[cyan]Appel LLM — mode={mode or 'auto'}..."):
        result = asyncio.run(manager.handle(req))

    if pretty:
        payload = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
        console.print(Syntax(payload, "json", theme="monokai"))
    else:
        if result.error:
            console.print(f"[bold red]Error ({result.provider}): {result.error}[/bold red]")
        else:
            console.print(f"[bold]{result.provider}[/bold] ({result.model}):")
            console.print(result.response)


if __name__ == "__main__":
    app()
