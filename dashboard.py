import re
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.text import Text
from rich import box

LOG_FILE = Path(__file__).parent / "training_log.txt"
TOTAL_EPOCHS = 50
UPDATE_INTERVAL = 3  # seconds

def parse_log(path):
    epochs_done = []
    current_epoch = None
    current_iter = 0
    current_total = 749
    current_acc1 = 0.0
    current_acc5 = 0.0
    current_loss = 0.0

    if not path.exists():
        return epochs_done, current_epoch, current_iter, current_total, current_acc1, current_acc5, current_loss

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines:
        # Completed epoch summary: * Acc@1 72.780 Acc@5 91.238
        m = re.search(r"\* Acc@1 ([\d.]+) Acc@5 ([\d.]+)", line)
        if m:
            epochs_done.append({
                "epoch": len(epochs_done) + 1,
                "acc1": float(m.group(1)),
                "acc5": float(m.group(2)),
            })

        # In-progress iter: Epoch: [7/50][100/749] LR: ... Loss x (avg) Acc@1 x (avg) Acc@5 x (avg)
        m = re.search(
            r"Epoch: \[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*Loss [\d.]+ \(([\d.]+)\).*Acc@1 [\d.]+ \(([\d.]+)\).*Acc@5 [\d.]+ \(([\d.]+)\)",
            line,
        )
        if m:
            current_epoch = int(m.group(1))
            current_iter = int(m.group(3))
            current_total = int(m.group(4))
            current_loss = float(m.group(5))
            current_acc1 = float(m.group(6))
            current_acc5 = float(m.group(7))

    return epochs_done, current_epoch, current_iter, current_total, current_acc1, current_acc5, current_loss


def build_display(epochs_done, current_epoch, current_iter, current_total, acc1, acc5, loss):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="table", ratio=2),
        Layout(name="stats", ratio=1),
    )

    # Header
    ep = current_epoch or (len(epochs_done))
    pct = round(100 * len(epochs_done) / TOTAL_EPOCHS)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    header_text = Text(
        f"  GLSim ViT-B/16 · CUB-200-2011  |  [{bar}] {pct}%  |  Época {ep}/{TOTAL_EPOCHS}",
        style="bold white on dark_blue",
        justify="center",
    )
    layout["header"].update(Panel(header_text, style="dark_blue"))

    # Epoch history table
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", show_edge=False)
    table.add_column("Época", justify="center", style="bold")
    table.add_column("Acc@1 treino", justify="right")
    table.add_column("Acc@5 treino", justify="right")
    table.add_column("Barra", justify="left", min_width=20)

    for e in epochs_done[-15:]:
        bar_len = int(e["acc1"] / 5)
        color = "green" if e["acc1"] >= 80 else "yellow" if e["acc1"] >= 60 else "red"
        acc_bar = f"[{color}]{'█' * bar_len}[/{color}]"
        table.add_row(
            str(e["epoch"]),
            f"[bold]{e['acc1']:.2f}%[/bold]",
            f"{e['acc5']:.2f}%",
            acc_bar,
        )

    layout["table"].update(Panel(table, title="[bold]Histórico de Épocas[/bold]", border_style="cyan"))

    # Stats panel
    best = max(epochs_done, key=lambda x: x["acc1"]) if epochs_done else None
    eta_min = round((TOTAL_EPOCHS - len(epochs_done)) * 7.5)

    stats_lines = []
    if current_epoch:
        iter_pct = int(100 * current_iter / current_total)
        iter_bar = "█" * (iter_pct // 5) + "░" * (20 - iter_pct // 5)
        stats_lines += [
            f"[bold yellow]Época atual:[/bold yellow] {current_epoch}/{TOTAL_EPOCHS}",
            f"[bold yellow]Iter:[/bold yellow] {current_iter}/{current_total}",
            f"[dim]{iter_bar}[/dim]",
            f"[bold yellow]Loss:[/bold yellow]  {loss:.4f}",
            f"[bold yellow]Acc@1:[/bold yellow] {acc1:.2f}%",
            f"[bold yellow]Acc@5:[/bold yellow] {acc5:.2f}%",
            "",
        ]
    if best:
        stats_lines += [
            f"[bold green]Melhor Acc@1:[/bold green] {best['acc1']:.2f}%",
            f"[bold green]Melhor época:[/bold green]  {best['epoch']}",
            "",
        ]
    if epochs_done:
        stats_lines.append(f"[dim]ETA: ~{eta_min} min[/dim]")
        stats_lines.append(f"[dim]Épocas concluídas: {len(epochs_done)}[/dim]")

    stats_text = Text.from_markup("\n".join(stats_lines) if stats_lines else "Aguardando início...")
    layout["stats"].update(Panel(stats_text, title="[bold]Status[/bold]", border_style="yellow"))

    # Footer
    layout["footer"].update(
        Panel(Text("  Ctrl+C para sair  |  Log: training_log.txt", style="dim", justify="center"))
    )

    return layout


def main():
    console = Console()
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            data = parse_log(LOG_FILE)
            epochs_done, cur_ep, cur_iter, cur_total, acc1, acc5, loss = data

            # Check if training finished
            finished = len(epochs_done) >= TOTAL_EPOCHS

            display = build_display(epochs_done, cur_ep, cur_iter, cur_total, acc1, acc5, loss)
            live.update(display)

            if finished:
                time.sleep(2)
                break

            time.sleep(UPDATE_INTERVAL)

    console.print("\n[bold green]Treinamento concluído![/bold green]")
    if epochs_done:
        best = max(epochs_done, key=lambda x: x["acc1"])
        console.print(f"Melhor Acc@1: [bold]{best['acc1']:.2f}%[/bold] (época {best['epoch']})")


if __name__ == "__main__":
    main()
