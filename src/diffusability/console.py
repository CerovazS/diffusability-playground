from rich.console import Console

_console = Console(force_terminal=True)

def ok(message: str) -> None:
    _console.print(f"[bold green]OK[/bold green] {message}")

def info(message: str) -> None:
    _console.print(f"[cyan]INFO[/cyan] {message}")

def warn(message: str) -> None:
    _console.print(f"[bold yellow]WARN[/bold yellow] {message}")

def error(message: str) -> None:
    _console.print(f"[bold red]ERROR[/bold red] {message}")
