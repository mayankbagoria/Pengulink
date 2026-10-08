"""pengulink.cli.main
~~~~~~~~~~~~~~~~~~~
Interactive Debug CLI for PenguLink daemon powered by Click and Rich.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pengulink.core.config import Config
from pengulink.core.exceptions import (
    DaemonAlreadyRunningError,
    DaemonNotRunningError,
    IPCError,
)
from pengulink.daemon.ipc import PenguLinkIPCClient, resolve_socket_path as ipc_resolve_socket_path

console = Console()
err_console = Console(stderr=True)


# =====================================================================
# Formatting Helpers
# =====================================================================

def format_state_badge(state: str) -> str:
    """Format an 8-state DeviceState string into a colored Rich badge."""
    state_upper = (state or "").upper()
    badges = {
        "CONNECTED": "[bold green]● CONNECTED[/]",
        "CONNECTING": "[bold yellow]◌ CONNECTING[/]",
        "PAIRING": "[bold blue]◌ PAIRING[/]",
        "PAIRED": "[blue]● PAIRED[/]",
        "DISCOVERED": "[cyan]● DISCOVERED[/]",
        "DISCONNECTING": "[dim yellow]○ DISCONNECTING[/]",
        "DISCONNECTED": "[dim]○ DISCONNECTED[/]",
        "ERROR": "[bold red]✕ ERROR[/]",
    }
    return badges.get(state_upper, f"[white]{state}[/]")


def format_battery_badge(battery: Optional[Dict[str, Any]]) -> str:
    """Format battery percentage and charging status."""
    if not battery:
        return "[dim]N/A[/]"

    charge = battery.get("current_charge") if battery.get("current_charge") is not None else -1
    is_charging = bool(battery.get("is_charging", False))

    if charge < 0:
        return "[dim]AC / Desktop[/]"

    if is_charging:
        return f"{charge}% [bold yellow]⚡ (Charging)[/]"
    elif charge <= 15:
        return f"{charge}% [bold red]🪫 (Low)[/]"
    return f"{charge}% [green]🔋[/]"


def format_otp_badge(auth_code: Optional[str]) -> str:
    """Format SMS/2FA verification codes prominently."""
    if not auth_code:
        return "[dim]-[/]"
    return f"[bold black on yellow]  {auth_code}  [/]"


def format_paired_badge(is_paired: bool) -> str:
    """Format paired boolean flag."""
    return "[bold green]Yes[/]" if is_paired else "[dim]No[/]"


def format_signal_bars(signal_strength: int) -> str:
    """Format coarse signal strength level (0..4) into Rich indicator."""
    bars = {
        4: "[bold green]●●●●[/]  (Great)",
        3: "[bold green]●●●○[/]  (Good)",
        2: "[bold yellow]●●○○[/]  (Moderate)",
        1: "[bold red]●○○○[/]  (Poor)",
        0: "[dim]○○○○[/]  (No Signal)",
    }
    return bars.get(signal_strength, "[dim]○○○○[/]  (No Signal)")


def format_network_badge(network_type: str) -> str:
    """Format cellular radio generation into a colored badge."""
    net_upper = (network_type or "Unknown").upper()
    if net_upper == "5G":
        return "[bold magenta]5G[/]"
    elif net_upper in ("LTE", "4G"):
        return "[bold cyan]LTE[/]"
    elif net_upper in ("3G", "HSPA", "HSPAP", "UMTS", "CDMA2000"):
        return "[yellow]3G[/]"
    elif net_upper in ("2G", "EDGE", "GPRS", "GSM", "CDMA", "IDEN"):
        return "[dim yellow]2G[/]"
    return f"[dim]{network_type}[/]"


# =====================================================================
# IPC Invocation & Error Handling Bridge
# =====================================================================

def resolve_socket_path(ctx: click.Context) -> Path:
    """Resolve Unix domain socket path using CLI options, config, or fallbacks."""
    custom_sock = ctx.obj.get("socket_path") if ctx.obj else None
    if custom_sock:
        return Path(custom_sock)

    config_path = ctx.obj.get("config_path") if ctx.obj else None
    try:
        cfg = Config.load(config_path=config_path)
        primary_sock = cfg.storage.ipc_socket_path
        fallback_sock = cfg.storage.config_dir / cfg.storage.ipc_socket_filename
        if not primary_sock.exists() and fallback_sock.exists():
            return fallback_sock
        return primary_sock
    except Exception:
        return ipc_resolve_socket_path()


def run_ipc_call(
    ctx: click.Context,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Any:
    """Execute a synchronous call to the daemon via PenguLinkIPCClient with standardized error handling."""
    socket_path = resolve_socket_path(ctx)
    client = PenguLinkIPCClient(socket_path=socket_path)

    try:
        return client.call_sync(method=method, params=params, timeout=timeout)
    except DaemonNotRunningError:
        console.print(
            "PenguLink daemon is not running. Start with 'pengulink daemon run' or start the background service."
        )
        sys.exit(1)
    except IPCError as exc:
        console.print(f"[bold red]IPC Error:[/] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]Unexpected Error:[/] {exc}")
        sys.exit(1)


# =====================================================================
# Root CLI Group
# =====================================================================

@click.group(name="pengulink")
@click.version_option(version="0.1.0", prog_name="pengulink")
@click.option(
    "--socket-path",
    "-s",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Explicit path to PenguLink daemon Unix domain socket.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Explicit path to pengulink config.json.",
)
@click.option(
    "--json-output",
    "--json",
    is_flag=True,
    default=False,
    help="Output machine-readable JSON instead of Rich formatting.",
)
@click.pass_context
def cli(ctx: click.Context, socket_path: Optional[Path], config: Optional[Path], json_output: bool) -> None:
    """PenguLink — KDE Connect Linux Daemon & Debug CLI."""
    ctx.ensure_object(dict)
    ctx.obj["socket_path"] = socket_path
    ctx.obj["config_path"] = config
    ctx.obj["json_output"] = json_output


# =====================================================================
# Device Discovery and Status Commands
# =====================================================================

@cli.command(name="discover", help="Trigger active UDP discovery and display peers.")
@click.option("--timeout", "-t", type=float, default=3.0, show_default=True, help="Sweep duration (seconds).")
@click.pass_context
def discover(ctx: click.Context, timeout: float) -> None:
    """Trigger UDP discovery sweep and display discovered peers in a Rich table."""
    with console.status(f"[bold cyan]Broadcasting UDP discovery on port 1716 ({timeout}s)...[/]"):
        peers = run_ipc_call(ctx, "discover", {"timeout": timeout, "refresh": True}, timeout=timeout + 5.0)

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(peers, indent=2))
        return

    if not peers:
        console.print("[yellow]No KDE Connect peers discovered on local network.[/]")
        return

    table = Table(title="Discovered KDE Connect Peers", box=box.ROUNDED)
    table.add_column("Device ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold white")
    table.add_column("Device Type", style="magenta")
    table.add_column("IP Address", style="green")
    table.add_column("Port", style="yellow")
    table.add_column("Paired", style="bold")

    for p in peers:
        table.add_row(
            p.get("device_id", ""),
            p.get("device_name", "Unknown"),
            p.get("device_type", "desktop"),
            p.get("sender_ip", p.get("ip", "")),
            str(p.get("tcp_port", p.get("port", 1716))),
            format_paired_badge(p.get("is_paired", False)),
        )
    console.print(table)


@cli.command(name="devices", help="Display known and paired devices.")
@click.pass_context
def devices(ctx: click.Context) -> None:
    """Display known/paired devices in a Rich table with state and battery levels."""
    dev_list = run_ipc_call(ctx, "get_devices", {})

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(dev_list, indent=2))
        return

    if not dev_list:
        console.print("[dim]No devices known or paired yet. Run 'pengulink discover' to find peers.[/]")
        return

    table = Table(title="Known & Paired Devices", box=box.ROUNDED)
    table.add_column("Device ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold white")
    table.add_column("Type", style="magenta")
    table.add_column("State", style="bold")
    table.add_column("Paired", style="bold")
    table.add_column("Battery")

    for dev in dev_list:
        table.add_row(
            dev.get("device_id", ""),
            dev.get("device_name", "Unknown"),
            dev.get("device_type", "desktop"),
            format_state_badge(dev.get("state", "DISCONNECTED")),
            format_paired_badge(dev.get("is_paired", False)),
            format_battery_badge(dev.get("battery")),
        )
    console.print(table)


# =====================================================================
# Pairing and Connection Management Commands
# =====================================================================

@cli.command(name="pair", help="Initiate pairing handshake with a remote device.")
@click.argument("device_id", type=str)
@click.option("--timeout", "-t", type=float, default=30.0, show_default=True, help="Pairing timeout (seconds).")
@click.pass_context
def pair(ctx: click.Context, device_id: str, timeout: float) -> None:
    """Initiate pairing, render 8-character verification code in a Rich panel, and await outcome."""
    result = run_ipc_call(
        ctx,
        "pair_device",
        {"device_id": device_id, "timeout": timeout, "wait": False},
        timeout=timeout + 5.0,
    )

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(result, indent=2))
        return

    verification_code = result.get("verification_code", "")
    device_name = result.get("device_name", device_id)
    status = result.get("status", "pending")

    if verification_code:
        panel_content = (
            f"\n[bold]Verification Code:[/] [bold yellow on black]  {verification_code}  [/]\n\n"
            f"[cyan]Please confirm that this 8-character verification code matches the prompt\n"
            f"on device [bold white]'{device_name}'[/] and accept the pairing request.[/]\n"
        )
        console.print(
            Panel(
                panel_content,
                title="[bold cyan]Pairing Verification[/]",
                border_style="cyan",
                box=box.ROUNDED,
                expand=False,
            )
        )

    if status == "paired":
        console.print(f"[bold green]✓ Pairing successful with {device_id} ({device_name})![/]")
    elif status == "rejected":
        console.print(f"[bold red]✗ Pairing rejected by device '{device_id}'.[/]")
        sys.exit(1)
    elif status == "timeout":
        console.print(f"[bold red]✗ Pairing timed out waiting for device '{device_id}'.[/]")
        sys.exit(1)
    else:
        # Await confirmation by polling devices status
        with console.status("[bold cyan]Waiting for remote device confirmation...[/]"):
            start_t = time.time()
            paired = False
            while time.time() - start_t < timeout:
                time.sleep(1.0)
                try:
                    devs = run_ipc_call(ctx, "get_devices", {})
                    target = next((d for d in devs if d.get("device_id") == device_id), None)
                    if target and target.get("is_paired"):
                        paired = True
                        break
                    if target and target.get("state") in ("ERROR", "DISCONNECTED"):
                        break
                except Exception:
                    pass

            if paired:
                console.print(f"[bold green]✓ Pairing successful with {device_id} ({device_name})![/]")
            else:
                console.print(f"[bold red]✗ Pairing negotiation ended without confirmation for '{device_id}'.[/]")
                sys.exit(1)


@cli.command(name="unpair", help="Unpair a device and revoke trust.")
@click.argument("device_id", type=str)
@click.option("--yes", "-y", is_flag=True, default=False, help="Confirm without prompting.")
@click.pass_context
def unpair(ctx: click.Context, device_id: str, yes: bool) -> None:
    """Unpair device and delete credentials from storage."""
    if not yes:
        click.confirm(f"Are you sure you want to unpair device '{device_id}'?", abort=True)

    result = run_ipc_call(ctx, "unpair_device", {"device_id": device_id})

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(result, indent=2))
        return

    console.print(f"[bold green]✓ Device '{device_id}' successfully unpaired.[/]")


@cli.command(name="connect", help="Trigger manual connection to a device.")
@click.argument("device_id", type=str)
@click.option("--timeout", "-t", type=float, default=10.0, show_default=True, help="Connection timeout.")
@click.pass_context
def connect(ctx: click.Context, device_id: str, timeout: float) -> None:
    """Manually initiate connection to a device."""
    with console.status(f"[bold cyan]Connecting to device '{device_id}'...[/]"):
        result = run_ipc_call(
            ctx,
            "connect_device",
            {"device_id": device_id, "timeout": timeout},
            timeout=timeout + 5.0,
        )

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(result, indent=2))
        return

    console.print(f"[bold green]✓ Connected to device '{device_id}'.[/]")


@cli.command(name="ping", help="Send a KDE Connect ping packet to a device.")
@click.argument("device_id", type=str)
@click.option("--message", "-m", type=str, default="Ping!", show_default=True, help="Ping message.")
@click.pass_context
def ping(ctx: click.Context, device_id: str, message: str) -> None:
    """Transmit a ping packet and confirm delivery."""
    result = run_ipc_call(ctx, "send_ping", {"device_id": device_id, "message": message})

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(result, indent=2))
        return

    console.print(f"[bold green]✓ Ping successfully sent to '{device_id}'.[/]")


# =====================================================================
# Notification Commands
# =====================================================================

@cli.command(name="notifications", help="List active notifications and highlight 2FA/OTP codes.")
@click.argument("device_id", required=False, default=None)
@click.option("--otp-only", is_flag=True, default=False, help="Only show notifications with OTP codes.")
@click.pass_context
def notifications(ctx: click.Context, device_id: Optional[str], otp_only: bool) -> None:
    """List notifications with highlighted OTP codes."""
    params = {}
    if device_id:
        params["device_id"] = device_id
    notifs = run_ipc_call(ctx, "get_notifications", params)

    if otp_only:
        notifs = [n for n in notifs if n.get("auth_code")]

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(notifs, indent=2))
        return

    if not notifs:
        console.print("[dim]No active notifications found.[/]")
        return

    table = Table(title="Active Mobile Notifications", box=box.ROUNDED)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Device", style="cyan")
    table.add_column("App", style="bold magenta")
    table.add_column("Title & Content", style="white")
    table.add_column("2FA / OTP Code", justify="center")
    table.add_column("Reply ID", style="dim cyan")
    table.add_column("Time", style="dim")

    for n in notifs:
        content_preview = f"[bold]{n.get('title', '')}[/]\n{n.get('text', '')}"
        table.add_row(
            n.get("notification_id", ""),
            (n.get("device_id") or "")[:8],
            n.get("app_name", "Unknown"),
            content_preview.strip(),
            format_otp_badge(n.get("auth_code")),
            n.get("request_reply_id") or "-",
            n.get("time_str", ""),
        )
    console.print(table)


@cli.command(name="dismiss", help="Dismiss a mobile notification.")
@click.argument("device_id", type=str)
@click.argument("notification_id", type=str)
@click.pass_context
def dismiss(ctx: click.Context, device_id: str, notification_id: str) -> None:
    """Dismiss a remote notification."""
    result = run_ipc_call(
        ctx,
        "dismiss_notification",
        {"device_id": device_id, "notification_id": notification_id},
    )

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(result, indent=2))
        return

    console.print(f"[bold green]✓ Notification '{notification_id}' dismissed on device '{device_id}'.[/]")


@cli.command(name="reply", help="Reply to a mobile notification.")
@click.argument("device_id", type=str)
@click.argument("reply_id", type=str)
@click.argument("message", type=str)
@click.pass_context
def reply(ctx: click.Context, device_id: str, reply_id: str, message: str) -> None:
    """Transmit quick reply to a messaging notification."""
    result = run_ipc_call(
        ctx,
        "reply_notification",
        {"device_id": device_id, "reply_id": reply_id, "message": message},
    )

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(result, indent=2))
        return

    console.print(f"[bold green]✓ Reply sent to device '{device_id}': \"{message}\"[/]")


@cli.command(name="connectivity", help="Display cellular connectivity report for a device.")
@click.argument("device_id", type=str)
@click.option("-r", "--request", is_flag=True, default=False, help="Send an on-demand refresh query to the device first.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def connectivity(ctx: click.Context, device_id: str, request: bool, json_flag: bool) -> None:
    """Display cellular connectivity report for a device."""
    is_json = json_flag or ctx.obj.get("json_output", False)

    if request:
        run_ipc_call(
            ctx,
            "request_connectivity_report",
            {"device_id": device_id},
        )

    report = run_ipc_call(
        ctx,
        "get_connectivity_report",
        {"device_id": device_id},
    )

    if is_json:
        click.echo(json.dumps(report, indent=2))
        return

    if not report:
        console.print(
            f"No connectivity report available yet for device '{device_id}'.\n"
            "Use '--request' to query the device."
        )
        return

    signal_strengths = report.get("signal_strengths") or {}
    ts = report.get("timestamp")
    time_info = ""
    if ts:
        elapsed = max(0, int(time.time() - ts))
        time_info = f"Updated: {elapsed} seconds ago"

    if not signal_strengths:
        content = (
            f"Device: {device_id}\n\n"
            "⚠ No active cellular subscriptions reported.\n"
            "  (Device may be in Airplane Mode or operating without SIM cards)"
        )
        console.print(
            Panel(
                content,
                title=f"[bold cyan]Cellular Connectivity: {device_id}[/]",
                subtitle=time_info,
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
        return

    table = Table(box=box.ROUNDED)
    table.add_column("SIM / Subscription", style="bold white")
    table.add_column("Network Type", justify="center")
    table.add_column("Signal Level", style="white")
    table.add_column("Status", style="green")

    for idx, (sub_id, data) in enumerate(signal_strengths.items()):
        net_type = data.get("network_type") or data.get("networkType") or "Unknown"
        sig_str = data.get("signal_strength") if data.get("signal_strength") is not None else data.get("signalStrength", 0)
        table.add_row(
            f"SIM {idx + 1} (Slot {sub_id})",
            format_network_badge(net_type),
            format_signal_bars(int(sig_str)),
            "Active",
        )

    console.print(
        Panel(
            table,
            title=f"[bold cyan]Cellular Connectivity: {device_id}[/]",
            subtitle=time_info,
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


# =====================================================================
# Daemon Management Subcommands
# =====================================================================

@cli.group(name="daemon", help="Manage or inspect PenguLink background daemon.")
def daemon() -> None:
    """Daemon lifecycle management."""
    pass


@daemon.command(name="run", help="Run PenguLink daemon in the foreground.")
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), default=None, help="Config path.")
@click.option("--socket-path", "-s", type=click.Path(path_type=Path), default=None, help="Socket path.")
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Logging level.",
)
def daemon_run(config: Optional[Path], socket_path: Optional[Path], log_level: Optional[str]) -> None:
    """Execute daemon process in the foreground until interrupted."""
    from pengulink.daemon.service import PenguLinkDaemon

    overrides: Dict[str, Any] = {}
    if socket_path:
        overrides.setdefault("storage", {})["ipc_socket_filename"] = socket_path.name
        overrides.setdefault("storage", {})["runtime_dir"] = socket_path.parent
    if log_level:
        overrides.setdefault("logging", {})["level"] = log_level.upper()

    cfg = Config.load(config_path=config, overrides=overrides)
    cfg.ensure_directories()

    daemon_inst = PenguLinkDaemon(config=cfg, socket_path=socket_path)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler() -> None:
        console.print("\n[yellow]Shutdown signal received. Stopping PenguLink daemon...[/]")
        loop.create_task(daemon_inst.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except (NotImplementedError, RuntimeError):
            pass

    console.print(f"[bold green]Starting PenguLink daemon (Device ID: {cfg.device_id[:8]}...)[/]")
    console.print(f"[cyan]Listening on IPC socket:[/] {daemon_inst.socket_path}")

    try:
        loop.run_until_complete(daemon_inst.start())
        loop.run_until_complete(daemon_inst.wait_until_stopped())
    except DaemonAlreadyRunningError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(daemon_inst.stop())
        loop.close()
        console.print("[green]PenguLink daemon stopped cleanly.[/]")


@daemon.command(name="status", help="Check if PenguLink daemon is running.")
@click.pass_context
def daemon_status(ctx: click.Context) -> None:
    """Check daemon IPC responsiveness and display status."""
    socket_path = resolve_socket_path(ctx)
    client = PenguLinkIPCClient(socket_path=socket_path)

    try:
        status_info = client.call_sync("status", timeout=3.0)
    except (DaemonNotRunningError, IPCError, Exception):
        console.print("[bold red]Status:[/] [bold red]● STOPPED (Inactive)[/]")
        console.print(
            "PenguLink daemon is not running. Start with 'pengulink daemon run' or start the background service."
        )
        sys.exit(1)

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(status_info, indent=2))
        return

    content = (
        f"[bold]Status:[/]             [bold green]● RUNNING (Active)[/]\n"
        f"[bold]Socket Path:[/]        {socket_path}\n"
        f"[bold]Local Device ID:[/]    {status_info.get('device_id', 'Unknown')}\n"
        f"[bold]Device Name:[/]        {status_info.get('device_name', 'Unknown')}\n"
        f"[bold]Connected Peers:[/]    {status_info.get('connected_devices', 0)}\n"
        f"[bold]Discovered Peers:[/]   {status_info.get('discovered_devices', 0)}\n"
        f"[bold]Known Devices:[/]      {status_info.get('known_devices', 0)}\n"
        f"[bold]Uptime:[/]             {status_info.get('uptime', 'Unknown')}\n"
    )
    console.print(
        Panel(
            content.strip(),
            title="[bold cyan]PenguLink Daemon Status[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


# =====================================================================
# Logs Command
# =====================================================================

@cli.command(name="logs", help="Fetch and display recent daemon log records.")
@click.option("--limit", "-n", type=int, default=50, show_default=True, help="Number of records to fetch.")
@click.option(
    "--level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Filter minimum log level.",
)
@click.pass_context
def logs(ctx: click.Context, limit: int, level: Optional[str]) -> None:
    """Fetch and display recent daemon log records in a formatted Rich table."""
    params: Dict[str, Any] = {"limit": limit}
    if level:
        params["level"] = level.upper()

    log_entries = run_ipc_call(ctx, "get_logs", params)

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(log_entries, indent=2))
        return

    if not log_entries:
        console.print("[dim]No log entries recorded yet.[/]")
        return

    table = Table(title="Recent Daemon Log Records", box=box.ROUNDED)
    table.add_column("Timestamp", style="dim")
    table.add_column("Level", style="bold")
    table.add_column("Logger", style="cyan")
    table.add_column("Message", style="white")

    level_styles = {
        "DEBUG": "[dim cyan]DEBUG[/]",
        "INFO": "[blue]INFO[/]",
        "WARNING": "[bold yellow]WARN[/]",
        "ERROR": "[bold red]ERROR[/]",
        "CRITICAL": "[bold white on red]CRIT[/]",
    }

    for entry in log_entries:
        lvl = str(entry.get("level", "INFO")).upper()
        styled_lvl = level_styles.get(lvl, f"[white]{lvl}[/]")
        table.add_row(
            entry.get("time_str", entry.get("timestamp_str", "")),
            styled_lvl,
            entry.get("logger", "pengulink"),
            entry.get("message", ""),
        )
    console.print(table)


if __name__ == "__main__":
    cli()
