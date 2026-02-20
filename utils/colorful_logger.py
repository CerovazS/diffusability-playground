from rich.console import Console

__all__ = ["warn", "info", "error", "ok"]

console = Console()

def warn(message: str):
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
    
def info(message: str):
    console.print(f"[bold blue]Info:[/bold blue] {message}")
    
def error(message: str):
    console.print(f"[bold red]Error:[/bold red] {message}")

def ok(message: str):
    console.print(f"[bold green]Success:[/bold green] {message}")
