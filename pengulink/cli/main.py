"""pengulink.cli.main
~~~~~~~~~~~~~~~~~~~
Interactive Debug CLI for PenguLink daemon powered by Click and Rich.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich import box
from rich.console import Console
from rich.markup import escape
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


def format_call_state_badge(state: str) -> str:
    """Format a Telephony CallState string into a colored Rich badge."""
    state_upper = (state or "").upper()
    badges = {
        "RINGING": "[bold yellow]🔔 RINGING[/]",
        "TALKING": "[bold green]📞 TALKING[/]",
        "MISSED": "[bold red]✕ MISSED[/]",
        "IDLE": "[dim]○ IDLE[/]",
    }
    return badges.get(state_upper, f"[white]{escape(state)}[/]")


def format_call_duration(duration_seconds: float | int | None) -> str:
    """Format call duration in seconds to standard MM:SS or HH:MM:SS format safely."""
    if duration_seconds is None:
        return "00:00"
    try:
        val = float(duration_seconds)
    except (TypeError, ValueError):
        return "00:00"

    if math.isnan(val) or math.isinf(val) or not math.isfinite(val) or val <= 0.0:
        return "00:00"

    try:
        total_seconds = int(val)
    except (ValueError, OverflowError):
        return "00:00"

    mins, secs = divmod(total_seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


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


def is_json_requested(ctx: Optional[click.Context]) -> bool:
    """Determine whether machine-readable JSON output was requested via root or subcommand flags."""
    curr = ctx
    while curr is not None:
        if curr.obj and isinstance(curr.obj, dict) and curr.obj.get("json_output"):
            return True
        if hasattr(curr, "params") and isinstance(curr.params, dict):
            if (
                curr.params.get("json_flag")
                or curr.params.get("json_output")
                or curr.params.get("json")
            ):
                return True
        curr = curr.parent
    if any(arg in sys.argv for arg in ("--json", "--json-output")):
        return True
    return False


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
    except DaemonNotRunningError as exc:
        if is_json_requested(ctx):
            click.echo(
                json.dumps(
                    {
                        "error": str(exc),
                        "code": getattr(exc, "code", -1),
                        "success": False,
                    },
                    indent=2,
                )
            )
        else:
            console.print(
                "PenguLink daemon is not running. Start with 'pengulink daemon run' or start the background service."
            )
        sys.exit(1)
    except IPCError as exc:
        if is_json_requested(ctx):
            err_code = getattr(exc, "code", None)
            if err_code is None and hasattr(exc, "context") and isinstance(exc.context, dict):
                err_code = exc.context.get("code")
            if err_code is None:
                err_code = -1
            click.echo(
                json.dumps(
                    {
                        "error": str(exc),
                        "code": err_code,
                        "success": False,
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"[bold red]IPC Error:[/] {exc}")
        sys.exit(1)
    except Exception as exc:
        if is_json_requested(ctx):
            click.echo(
                json.dumps(
                    {
                        "error": str(exc),
                        "code": getattr(exc, "code", -1),
                        "success": False,
                    },
                    indent=2,
                )
            )
        else:
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

class PairGroup(click.Group):
    """Click group for pairing operations that defaults to initiating outbound pairing if no subcommand is given."""

    def resolve_command(self, ctx: click.Context, args: list) -> tuple:
        cmd_name = click.utils.make_str(args[0]) if args else None
        if cmd_name in self.commands:
            return super().resolve_command(ctx, args)
        if not args or args[0] in ("--help", "-h"):
            return super().resolve_command(ctx, args)
        cmd = self.get_command(ctx, "request")
        return "request", cmd, args

    def parse_args(self, ctx: click.Context, args: list) -> list:
        if not args or (args[0] not in self.commands and args[0] not in ("--help", "-h")):
            args = ["request"] + list(args)
        return super().parse_args(ctx, args)


@cli.group(name="pair", cls=PairGroup, invoke_without_command=False, help="Initiate or manage pairing handshake with remote devices.")
def pair() -> None:
    """Pairing command group for initiating, accepting, or rejecting device pairing."""
    pass


@pair.command(name="request", help="Initiate pairing handshake with a remote device.")
@click.argument("device_id", type=str)
@click.option("--timeout", "-t", type=float, default=30.0, show_default=True, help="Pairing timeout (seconds).")
@click.pass_context
def pair_request(ctx: click.Context, device_id: str, timeout: float) -> None:
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


@pair.command(name="accept", help="Accept an incoming pairing request from a remote device.")
@click.argument("device_id", type=str)
@click.pass_context
def pair_accept(ctx: click.Context, device_id: str) -> None:
    """Accept an inbound pairing request from device_id."""
    result = run_ipc_call(
        ctx,
        "pair_accept",
        {"device_id": device_id},
    )
    if ctx.obj.get("json_output"):
        click.echo(json.dumps(result, indent=2))
        return

    msg = result.get("message", f"Accepted pairing with {device_id}")
    console.print(f"[bold green]✓ {msg}[/]")


@pair.command(name="reject", help="Reject an incoming pairing request from a remote device.")
@click.argument("device_id", type=str)
@click.pass_context
def pair_reject(ctx: click.Context, device_id: str) -> None:
    """Reject an inbound pairing request from device_id."""
    result = run_ipc_call(
        ctx,
        "pair_reject",
        {"device_id": device_id},
    )
    if ctx.obj.get("json_output"):
        click.echo(json.dumps(result, indent=2))
        return

    msg = result.get("message", f"Rejected pairing from {device_id}")
    console.print(f"[bold yellow]✗ {msg}[/]")


@pair.command(name="requests", help="List active pending inbound pairing requests.")
@click.pass_context
def pair_requests(ctx: click.Context) -> None:
    """List pending inbound pairing requests awaiting user confirmation."""
    result = run_ipc_call(ctx, "get_pending_pairings", {})
    if ctx.obj.get("json_output"):
        click.echo(json.dumps(result, indent=2))
        return

    if not result:
        console.print("[dim]No pending inbound pairing requests.[/]")
        return

    table = Table(title="Pending Inbound Pairing Requests", box=box.ROUNDED)
    table.add_column("Device ID", style="cyan")
    table.add_column("Device Name", style="white")
    table.add_column("Verification Code", style="bold yellow")
    table.add_column("Action", style="green")

    for req in result:
        table.add_row(
            req.get("device_id", ""),
            req.get("device_name", ""),
            req.get("verification_code", ""),
            f"pengulink pair accept {req.get('device_id')}",
        )
    console.print(table)


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
@click.option("--host", "-h", type=str, default=None, help="Target host/IP address override.")
@click.option("--port", "-p", type=int, default=None, help="Target TCP port override.")
@click.pass_context
def connect(
    ctx: click.Context,
    device_id: str,
    timeout: float,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """Manually initiate connection to a device."""
    payload: Dict[str, Any] = {"device_id": device_id}
    if host is not None:
        payload["host"] = host
    if port is not None:
        payload["port"] = int(port)
    if timeout != 10.0:
        payload["timeout"] = timeout

    socket_path = resolve_socket_path(ctx)
    client = PenguLinkIPCClient(socket_path=socket_path)

    with console.status(f"[bold cyan]Connecting to device '{device_id}'...[/]"):
        try:
            result = client.call_sync("connect", payload)
        except DaemonNotRunningError as exc:
            if is_json_requested(ctx):
                click.echo(
                    json.dumps(
                        {
                            "error": str(exc),
                            "code": getattr(exc, "code", -1),
                            "success": False,
                        },
                        indent=2,
                    )
                )
            else:
                console.print(
                    "PenguLink daemon is not running. Start with 'pengulink daemon run' or start the background service."
                )
            sys.exit(1)
        except IPCError as exc:
            if is_json_requested(ctx):
                err_code = getattr(exc, "code", None)
                if err_code is None and hasattr(exc, "context") and isinstance(exc.context, dict):
                    err_code = exc.context.get("code")
                if err_code is None:
                    err_code = -1
                click.echo(
                    json.dumps(
                        {
                            "error": str(exc),
                            "code": err_code,
                            "success": False,
                        },
                        indent=2,
                    )
                )
            else:
                console.print(f"[bold red]IPC Error:[/] {exc}")
            sys.exit(1)
        except Exception as exc:
            if is_json_requested(ctx):
                click.echo(
                    json.dumps(
                        {
                            "error": str(exc),
                            "code": getattr(exc, "code", -1),
                            "success": False,
                        },
                        indent=2,
                    )
                )
            else:
                console.print(f"[bold red]Unexpected Error:[/] {exc}")
            sys.exit(1)

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

    _daemon_shutdown_tasks: set[asyncio.Task[Any]] = set()

    def _signal_handler() -> None:
        console.print("\n[yellow]Shutdown signal received. Stopping PenguLink daemon...[/]")
        task = loop.create_task(daemon_inst.stop())
        _daemon_shutdown_tasks.add(task)
        task.add_done_callback(_daemon_shutdown_tasks.discard)

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
    except (DaemonNotRunningError, IPCError, Exception) as exc:
        if is_json_requested(ctx):
            click.echo(
                json.dumps(
                    {
                        "error": str(exc),
                        "code": getattr(exc, "code", -1),
                        "success": False,
                    },
                    indent=2,
                )
            )
        else:
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


@daemon.command(name="stop", help="Stop running PenguLink background daemon.")
@click.pass_context
def daemon_stop(ctx: click.Context) -> None:
    """Request running daemon to stop cleanly via IPC."""
    socket_path = resolve_socket_path(ctx)
    client = PenguLinkIPCClient(socket_path=socket_path)
    try:
        res = client.call_sync("stop_daemon")
    except (DaemonNotRunningError, FileNotFoundError, ConnectionRefusedError) as exc:
        if is_json_requested(ctx):
            click.echo(
                json.dumps(
                    {
                        "error": str(exc),
                        "code": -1,
                        "success": False,
                    },
                    indent=2,
                )
            )
        else:
            console.print("[yellow]PenguLink daemon is not running.[/]")
        sys.exit(1)
    except Exception as exc:
        if is_json_requested(ctx):
            click.echo(
                json.dumps(
                    {
                        "error": str(exc),
                        "code": getattr(exc, "code", -1),
                        "success": False,
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"[bold red]Failed to stop daemon:[/] {exc}")
        sys.exit(1)

    if ctx.obj.get("json_output"):
        click.echo(json.dumps(res, indent=2))
        return

    console.print("[bold green]Daemon stopped successfully.[/]")


# =====================================================================
# Clipboard Management Commands
# =====================================================================

@cli.group(name="clipboard", help="Inspect and synchronize system clipboard with connected devices.")
def clipboard() -> None:
    """Clipboard synchronization management."""
    pass


@clipboard.command(name="get", help="Retrieve current clipboard text.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.option("--raw", "-r", is_flag=True, default=False, help="Output raw text without panel formatting.")
@click.pass_context
def clipboard_get(ctx: click.Context, json_flag: bool, raw: bool) -> None:
    """Output current clipboard text or JSON representation."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    result = run_ipc_call(ctx, "get_clipboard", {})

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    content = result.get("content", "") if isinstance(result, dict) else ""
    if not content and not raw:
        console.print("[dim]Clipboard is empty.[/]")
        return

    # If piped/redirected or --raw requested, output exact unadorned text
    if raw or not sys.stdout.isatty():
        click.echo(content, nl=False)
        return

    # Rich panel presentation for interactive terminal
    ts = result.get("timestamp") if isinstance(result, dict) else None
    time_info = ""
    if ts and ts > 0:
        elapsed = max(0, int(time.time() - (ts / 1000.0)))
        time_info = f"Updated: {elapsed}s ago"

    chars_info = f"{len(content)} chars"
    subtitle = f"{time_info} ({chars_info})" if time_info else chars_info

    console.print(
        Panel(
            content,
            title="[bold cyan]Clipboard Content[/]",
            subtitle=subtitle,
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


@clipboard.command(name="set", help="Update clipboard locally and broadcast to connected peers.")
@click.argument("text", type=str, required=False, default=None)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def clipboard_set(ctx: click.Context, text: Optional[str], json_flag: bool) -> None:
    """Update clipboard locally and broadcast to connected devices."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))

    # Support reading from stdin if text omitted or '-'
    if text is None or text == "-":
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read()
            if text is None and not stdin_content:
                raise click.UsageError("Missing argument 'TEXT' or piped stdin.")
            text = stdin_content
        elif text is None:
            raise click.UsageError("Missing argument 'TEXT' or piped stdin.")

    # Validate payload size (<= 120 KB = 120,000 bytes)
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) > 120_000:
        raise click.ClickException(
            f"Clipboard content exceeds maximum allowed size ({len(encoded):,} bytes > 120,000 bytes)"
        )

    result = run_ipc_call(ctx, "set_clipboard", {"content": text})

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    byte_count = result.get("bytes", len(encoded)) if isinstance(result, dict) else len(encoded)
    console.print(
        f"[bold green]✓ Clipboard updated ({byte_count} bytes) and broadcast to connected devices.[/]"
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


# =====================================================================
# Find My Phone Command
# =====================================================================

@cli.command(name="find-phone", help="Trigger high-priority ring on a connected Android device.")
@click.argument("device_id", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def find_phone(ctx: click.Context, device_id: str, json_flag: bool) -> None:
    """Send a locate alert packet to a connected peer device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    result = run_ipc_call(ctx, "find_phone", {"device_id": device_id})

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    console.print(
        Panel(
            f"[bold]Locate request transmitted to device:[/] [bold cyan]{device_id}[/]\n\n"
            "[green]✓ The remote device is ringing with high-priority alarm sound.[/]\n"
            "[dim]Note: Repeat the command to toggle/cancel if supported by peer.[/]",
            title="[bold yellow]🔔 Find My Phone[/]",
            border_style="yellow",
            box=box.ROUNDED,
        )
    )


# =====================================================================
# Desktop Alert Commands
# =====================================================================

@cli.group(name="alert", help="Desktop alert and locate sound management.")
def alert() -> None:
    """Desktop alert management (Find This Device)."""
    pass


@alert.command(name="test", help="Trigger a local test of the desktop alert service.")
@click.option(
    "--duration",
    "-d",
    type=float,
    default=30.0,
    show_default=True,
    help="Maximum alert duration in seconds.",
)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def alert_test(ctx: click.Context, duration: float, json_flag: bool) -> None:
    """Test local alert audio playback and notification companion."""
    if duration <= 0:
        raise click.UsageError("Duration must be a positive number greater than 0.")

    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    result = run_ipc_call(ctx, "alert_test", {"duration": duration})

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    backend = result.get("backend", "unknown") if isinstance(result, dict) else "unknown"
    console.print(
        Panel(
            f"[bold]Desktop alert triggered locally.[/]\n\n"
            f"[cyan]Audio Backend:[/] {backend}\n"
            f"[cyan]Max Duration:[/]  {duration} seconds\n\n"
            "[dim]Run [bold]pengulink alert stop[/bold] to silence the alert.[/]",
            title="[bold red]🔊 Desktop Alert Test (Ringing)[/]",
            border_style="red",
            box=box.ROUNDED,
        )
    )


@alert.command(name="stop", help="Stop any active desktop alert sound immediately.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def alert_stop(ctx: click.Context, json_flag: bool) -> None:
    """Silence any active alert playback and clear status."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    result = run_ipc_call(ctx, "alert_stop", {})

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    stopped = result.get("stopped", False) if isinstance(result, dict) else False
    if stopped:
        console.print("[bold green]✓ Desktop alert stopped and silenced successfully.[/]")
    else:
        console.print("[dim]No desktop alert is currently active. Alert stopped/silenced.[/]")


@alert.command(name="status", help="Display current status of the desktop alert service.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def alert_status(ctx: click.Context, json_flag: bool) -> None:
    """Show whether the alert service is ringing, audio backend in use, and time remaining."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    result = run_ipc_call(ctx, "alert_status", {})

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    is_active = bool(result.get("active", False)) if isinstance(result, dict) else False
    source = result.get("source", "none") if isinstance(result, dict) else "none"
    backend = result.get("backend", "none") if isinstance(result, dict) else "none"
    remaining = result.get("remaining", result.get("remaining_seconds", 0.0)) if isinstance(result, dict) else 0.0

    table = Table(title="[bold cyan]Desktop Alert Service Status[/]", box=box.ROUNDED)
    table.add_column("Property", style="bold white")
    table.add_column("Value")

    state_badge = "[bold red]● RINGING (Active)[/]" if is_active else "[dim]○ INACTIVE (Idle)[/]"
    table.add_row("Status", state_badge)
    table.add_row("Source", f"[yellow]{source}[/]" if is_active else "[dim]None[/]")
    table.add_row("Audio Backend", f"[green]{backend}[/]")
    table.add_row("Remaining Time", f"{remaining:.1f}s" if is_active else "[dim]-[/]")

    console.print(table)


# =====================================================================
# Share Commands Group
# =====================================================================

@cli.group(name="share", help="Share files, text, and URLs with connected devices.")
def share() -> None:
    """Share files, text, and URLs with connected devices."""
    pass


@share.command(name="text", help="Share text snippet with a connected device.")
@click.argument("device_id", type=str)
@click.argument("text", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def share_text_cmd(ctx: click.Context, device_id: str, text: str, json_flag: bool) -> None:
    """Share text snippet with a remote device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    result = run_ipc_call(ctx, "share_text", {"device_id": device_id, "text": text})

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    snippet = (text[:60] + "...") if len(text) > 60 else text
    console.print(
        Panel(
            f"[bold]Target Device:[/] [bold cyan]{device_id}[/]\n"
            f"[bold]Text Content:[/]  {snippet}\n\n"
            "[green]✓ Text snippet transmitted and shared successfully.[/]",
            title="[bold blue]📤 Share Text[/]",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


@share.command(name="url", help="Share a web URL with a connected device.")
@click.argument("device_id", type=str)
@click.argument("url", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def share_url_cmd(ctx: click.Context, device_id: str, url: str, json_flag: bool) -> None:
    """Share a web URL with a remote device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    result = run_ipc_call(ctx, "share_url", {"device_id": device_id, "url": url})

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    console.print(
        Panel(
            f"[bold]Target Device:[/] [bold cyan]{device_id}[/]\n"
            f"[bold]Shared URL:[/]    [link={url}]{url}[/link]\n\n"
            "[green]✓ Web URL transmitted and shared successfully.[/]",
            title="[bold blue]🔗 Share URL[/]",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


@share.command(name="file", help="Share one or more files with a connected device.")
@click.argument("device_id", type=str)
@click.argument("file_paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--open", "open_flag", is_flag=True, default=False, help="Request remote device to open file upon receipt.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def share_file_cmd(ctx: click.Context, device_id: str, file_paths: Tuple[Path, ...], open_flag: bool, json_flag: bool) -> None:
    """Share one or more files with a remote device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    str_paths = [str(p.resolve()) for p in file_paths]
    params: Dict[str, Any] = {
        "device_id": device_id,
        "open": open_flag,
        "file_path": str_paths[0] if len(str_paths) == 1 else str_paths,
    }
    result = run_ipc_call(ctx, "share_file", params, timeout=120.0)

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    table = Table(title=f"Transferred Files to {device_id}", box=box.ROUNDED)
    table.add_column("Filename", style="bold white")
    table.add_column("Size", justify="right", style="cyan")
    table.add_column("Status", justify="center")

    for p in file_paths:
        sz = p.stat().st_size
        sz_str = f"{sz / 1024:.1f} KB" if sz < 1048576 else f"{sz / 1048576:.2f} MB"
        table.add_row(p.name, sz_str, "[bold green]Transferred[/]")

    console.print(table)


# =====================================================================
# Contacts Commands
# =====================================================================

def format_phone_type_badge(phone_type: str) -> str:
    """Format phone label into a colored Rich badge."""
    ptype = (phone_type or "CELL").upper()
    badges = {
        "CELL": "[bold cyan]CELL[/]",
        "MOBILE": "[bold cyan]MOB[/]",
        "WORK": "[bold blue]WORK[/]",
        "HOME": "[bold green]HOME[/]",
        "VOICE": "[bold magenta]VOICE[/]",
        "MAIN": "[bold purple]MAIN[/]",
        "OTHER": "[dim]OTHER[/]",
    }
    return badges.get(ptype, f"[dim]{ptype}[/]")


def format_email_type_badge(email_type: str) -> str:
    """Format email label into a colored Rich badge."""
    etype = (email_type or "HOME").upper()
    badges = {
        "HOME": "[bold yellow]HOME[/]",
        "WORK": "[bold blue]WORK[/]",
        "INTERNET": "[bold cyan]NET[/]",
        "OTHER": "[dim]OTHER[/]",
    }
    return badges.get(etype, f"[dim]{etype}[/]")


def format_contact_phones(phones: List[Dict[str, Any]]) -> str:
    """Format phone numbers with badges for table row display."""
    if not phones:
        return "[dim]-[/]"
    formatted = []
    for p in phones:
        if isinstance(p, dict):
            num = p.get("number", "")
            ptype = p.get("type", "CELL")
            badge = format_phone_type_badge(ptype)
            formatted.append(f"{badge} {num}")
        else:
            formatted.append(str(p))
    return "\n".join(formatted)


def format_contact_emails(emails: List[Dict[str, Any]]) -> str:
    """Format email addresses with badges for table row display."""
    if not emails:
        return "[dim]-[/]"
    formatted = []
    for e in emails:
        if isinstance(e, dict):
            addr = e.get("email", "")
            etype = e.get("type", "HOME")
            badge = format_email_type_badge(etype)
            formatted.append(f"{badge} {addr}")
        else:
            formatted.append(str(e))
    return "\n".join(formatted)


@cli.group(name="contacts", help="Manage and inspect synchronized contacts from connected devices.")
def contacts() -> None:
    """Contacts synchronization and address book management."""
    pass


@contacts.command(name="list", help="List synchronized contacts for a device.")
@click.argument("device_id", type=str)
@click.option("--limit", "-l", type=int, default=None, help="Maximum number of contacts to display.")
@click.option("--offset", "-o", type=int, default=0, help="Offset into contact list for pagination.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def contacts_list_cmd(ctx: click.Context, device_id: str, limit: Optional[int], offset: int, json_flag: bool) -> None:
    """List synchronized contacts for a paired or known device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")
    if offset < 0:
        raise click.UsageError("Option '--offset' must be non-negative.")
    if limit is not None and limit < 0:
        raise click.UsageError("Option '--limit' must be non-negative.")

    params: Dict[str, Any] = {"device_id": device_id, "offset": offset}
    if limit is not None:
        params["limit"] = limit

    result = run_ipc_call(ctx, "contacts_list", params)

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    contacts_data = result.get("contacts", [])
    total = result.get("total", len(contacts_data))

    if not contacts_data:
        console.print(f"[dim]No contacts found for device '{device_id}'.[/]")
        console.print(f"[dim]Run 'pengulink contacts sync {device_id}' to synchronize contacts from the device.[/]")
        return

    table = Table(
        title=f"Contacts for {device_id} (Showing {len(contacts_data)} of {total})",
        box=box.ROUNDED,
    )
    table.add_column("UID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold white")
    table.add_column("Phone Numbers", style="white")
    table.add_column("Emails", style="white")
    table.add_column("Organization", style="dim")

    for c in contacts_data:
        uid = str(c.get("uid", "-"))
        name = str(c.get("name") or "Unknown Contact")
        phones_str = format_contact_phones(c.get("phones", []))
        emails_str = format_contact_emails(c.get("emails", []))
        org = str(c.get("organization") or "-")
        table.add_row(uid, name, phones_str, emails_str, org)

    console.print(table)


@contacts.command(name="search", help="Search synchronized contacts by name, number, or email.")
@click.argument("device_id", type=str)
@click.argument("query", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def contacts_search_cmd(ctx: click.Context, device_id: str, query: str, json_flag: bool) -> None:
    """Search synchronized contacts by substring query."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    query = query.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")
    if not query:
        raise click.UsageError("Argument 'QUERY' must not be empty.")

    result = run_ipc_call(ctx, "contacts_search", {"device_id": device_id, "query": query})

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    matches = result.get("matches", [])
    if not matches:
        console.print(f"[dim]No contacts matching '{query}' found for device '{device_id}'.[/]")
        return

    table = Table(
        title=f"Search Results for '{query}' on {device_id} ({len(matches)} match(es))",
        box=box.ROUNDED,
    )
    table.add_column("UID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold white")
    table.add_column("Phone Numbers", style="white")
    table.add_column("Emails", style="white")
    table.add_column("Organization", style="dim")

    for c in matches:
        uid = str(c.get("uid", "-"))
        name = str(c.get("name") or "Unknown Contact")
        phones_str = format_contact_phones(c.get("phones", []))
        emails_str = format_contact_emails(c.get("emails", []))
        org = str(c.get("organization") or "-")
        table.add_row(uid, name, phones_str, emails_str, org)

    console.print(table)


@contacts.command(name="view", help="View detailed contact card or raw vCard.")
@click.argument("device_id", type=str)
@click.argument("uid", type=str)
@click.option("--raw", "-r", is_flag=True, default=False, help="Output verbatim RFC 2426 vCard text.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def contacts_view_cmd(ctx: click.Context, device_id: str, uid: str, raw: bool, json_flag: bool) -> None:
    """View detailed contact card or raw vCard representation."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    uid = uid.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")
    if not uid:
        raise click.UsageError("Argument 'UID' must not be empty.")

    result = run_ipc_call(ctx, "contacts_get", {"device_id": device_id, "uid": uid})

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    contact = result.get("contact", {})
    if raw:
        raw_vcard = contact.get("raw_vcard", "")
        if not raw_vcard:
            console.print("[dim]No raw vCard available for this contact record.[/]")
        else:
            click.echo(raw_vcard, nl=False)
        return

    name = contact.get("name") or "Unknown Contact"
    phones = contact.get("phones", [])
    emails = contact.get("emails", [])
    org = contact.get("organization")
    title = contact.get("title")
    note = contact.get("note")
    ts = contact.get("timestamp")
    cached_at = contact.get("cached_at")

    lines = [f"[bold]UID:[/] [cyan]{uid}[/]"]
    if org or title:
        org_display = f"{title} at {org}" if (title and org) else (org or title)
        lines.append(f"[bold]Organization:[/] {org_display}")

    if phones:
        lines.append("\n[bold]Phone Numbers:[/]")
        for p in phones:
            ptype = p.get("type", "CELL")
            num = p.get("number", "")
            norm = p.get("normalized", "")
            badge = format_phone_type_badge(ptype)
            norm_str = f" [dim]({norm})[/]" if norm and norm != num else ""
            lines.append(f"  • {badge} {num}{norm_str}")

    if emails:
        lines.append("\n[bold]Email Addresses:[/]")
        for e in emails:
            etype = e.get("type", "HOME")
            addr = e.get("email", "")
            badge = format_email_type_badge(etype)
            lines.append(f"  • {badge} {addr}")

    if note:
        lines.append(f"\n[bold]Notes:[/] {note}")

    sub_parts = []
    if ts and ts > 0:
        sub_parts.append(f"Remote timestamp: {int(ts)}")
    if cached_at and cached_at > 0:
        try:
            cached_dt = datetime.datetime.fromtimestamp(cached_at).strftime("%Y-%m-%d %H:%M:%S")
            sub_parts.append(f"Cached at: {cached_dt}")
        except Exception:
            pass
    subtitle = " | ".join(sub_parts) if sub_parts else None

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold cyan]👤 Contact: {name}[/]",
            subtitle=subtitle,
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


@contacts.command(name="sync", help="Synchronize contact address book with target device over TLS.")
@click.argument("device_id", type=str)
@click.option("--force", "-f", is_flag=True, default=False, help="Force full synchronization regardless of timestamp manifest.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def contacts_sync_cmd(ctx: click.Context, device_id: str, force: bool, json_flag: bool) -> None:
    """Synchronize contact address book with target device over active TLS stream."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    result = run_ipc_call(ctx, "contacts_sync", {"device_id": device_id, "force": force}, timeout=60.0)

    if is_json:
        click.echo(json.dumps(result, indent=2))
        return

    total = result.get("total", 0)
    added = result.get("added", 0)
    updated = result.get("updated", 0)
    deleted = result.get("deleted", 0)
    duration = result.get("duration", 0.0)

    summary_text = (
        f"[bold]Target Device:[/] [bold cyan]{device_id}[/]\n"
        f"[bold]Total Contacts:[/] [bold white]{total}[/]\n"
        f"[bold]Changes:[/] [green]+{added} added[/], [yellow]~{updated} updated[/], [red]-{deleted} deleted[/]\n"
        f"[bold]Duration:[/] [dim]{duration:.2f}s[/]\n\n"
        "[green]✓ Contact address book successfully synchronized and cached locally.[/]"
    )
    console.print(
        Panel(
            summary_text,
            title="[bold green]📇 Contacts Synchronization[/]",
            border_style="green",
            box=box.ROUNDED,
        )
    )


# =====================================================================
# MPRIS Media Control Commands
# =====================================================================

def format_time_ms(ms: int) -> str:
    """Format milliseconds into MM:SS or HH:MM:SS."""
    if ms < 0:
        return "--:--"
    total_seconds = ms // 1000
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_progress_bar(pos_ms: int, length_ms: int, width: int = 24) -> str:
    """Format position progress bar with percentage."""
    pos_str = format_time_ms(pos_ms)
    if length_ms <= 0:
        bar = "=" * (width // 2) + "-" * (width // 2)
        return f"{pos_str} [cyan][{bar}][/] (Live / Unknown duration)"

    len_str = format_time_ms(length_ms)
    ratio = min(1.0, max(0.0, pos_ms / length_ms)) if length_ms > 0 else 0.0
    pct = int(ratio * 100)
    filled = int(ratio * width)
    bar = "=" * max(0, filled - 1) + (">" if 0 < filled < width else "=" if filled == width else "")
    empty = "-" * (width - len(bar))
    return f"{pos_str} / {len_str} [cyan][{bar}{empty}][/] {pct}%"


def format_volume_bar(volume: int, width: int = 15) -> str:
    """Format volume percentage and bar."""
    vol = max(0, min(100, volume))
    filled = int((vol / 100) * width)
    empty = width - filled
    color = "green" if vol > 25 else "yellow" if vol > 10 else "red"
    return f"{vol}% [{color}][{'━' * filled}{'╸' if empty > 0 else ''}{'─' * max(0, empty - 1)}][/]"


def format_playback_badge(is_playing: bool) -> str:
    """Format playback state badge."""
    return "[bold green]▶ Playing[/]" if is_playing else "[bold yellow]⏸ Paused[/]"


@cli.group(name="mpris", help="Media player control and inspection for connected devices.")
def mpris() -> None:
    """Inspect and control remote media players on paired devices."""
    pass


@mpris.command(name="players", help="List available media players on target device.")
@click.argument("device_id", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_players_cmd(ctx: click.Context, device_id: str, json_flag: bool) -> None:
    """List available media players on target device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    res = run_ipc_call(ctx, "mpris_list_players", {"device_id": device_id})
    players = res.get("players", [])
    active = res.get("active_player")

    if is_json:
        click.echo(json.dumps(players, indent=2))
        return

    if not players:
        console.print(f"No active media players found for device '{escape(device_id)}'.")
        return

    table = Table(
        title=f"Media Players on [bold cyan]{escape(device_id)}[/]",
        box=box.ROUNDED,
    )
    table.add_column("Player", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Active", style="green")

    for p in players:
        is_act = "✓" if p == active else ""
        table.add_row(escape(p), "Available", is_act)

    console.print(table)


@mpris.command(name="status", help="Get media playback status and track metadata.")
@click.argument("device_id", type=str)
@click.option("--player", "-p", type=str, default=None, help="Target player name.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_status_cmd(ctx: click.Context, device_id: str, player: Optional[str], json_flag: bool) -> None:
    """Get media playback status and track metadata for active or selected player."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    params: Dict[str, Any] = {"device_id": device_id}
    if player:
        params["player"] = player.strip()

    res = run_ipc_call(ctx, "mpris_get_status", params)
    status = res.get("status")

    if is_json:
        click.echo(json.dumps(status or {}, indent=2))
        return

    if not status:
        console.print(f"No media player status available for device '{escape(device_id)}'.")
        return

    player_name = status.get("player") or "Unknown Player"
    safe_player = escape(player_name)
    safe_title = escape(status.get("title") or "Unknown Title")
    safe_artist = escape(status.get("artist") or "Unknown Artist")
    safe_album = escape(status.get("album") or "Unknown Album")
    is_playing = bool(status.get("isPlaying", False))
    pos = int(status.get("pos", 0))
    length = int(status.get("length", -1))
    vol = int(status.get("volume", 100))
    loop = escape(status.get("loopStatus") or "None")
    shuf = "On" if status.get("shuffle") else "Off"

    lines = [
        f"  {format_playback_badge(is_playing)}",
        "",
        f"  [bold]Title:[/]    {safe_title}",
        f"  [bold]Artist:[/]   {safe_artist}",
        f"  [bold]Album:[/]    {safe_album}",
        "",
        f"  [bold]Progress:[/] {format_progress_bar(pos, length)}",
        f"  [bold]Volume:[/]   {format_volume_bar(vol)}",
        "",
        "  [bold]Controls:[/] [bold cyan]⏪ Prev[/]  [bold cyan]⏩ Next[/]  [bold cyan]🔍 Seek[/]",
        f"  [bold]Loop:[/]     [dim]{loop}[/]     [bold]Shuffle:[/] [dim]{shuf}[/]",
    ]

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold cyan]🎵 Media Status: {safe_player} on {escape(device_id)}[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def _handle_transport_cmd(
    ctx: click.Context,
    device_id: str,
    action: str,
    player: Optional[str],
    json_flag: bool,
) -> None:
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    params: Dict[str, Any] = {"device_id": device_id, "action": action}
    if player:
        params["player"] = player.strip()

    res = run_ipc_call(ctx, "mpris_send_action", params)
    if is_json:
        click.echo(json.dumps(res, indent=2))
        return

    safe_p = escape(res.get("player") or "player")
    console.print(f"[green]✓ Action '{escape(action)}' sent to {safe_p} on '{escape(device_id)}'.[/]")


@mpris.command(name="play", help="Resume or start playback.")
@click.argument("device_id", type=str)
@click.option("--player", "-p", type=str, default=None, help="Target player name.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_play_cmd(ctx: click.Context, device_id: str, player: Optional[str], json_flag: bool) -> None:
    """Send Play action to media player."""
    _handle_transport_cmd(ctx, device_id, "Play", player, json_flag)


@mpris.command(name="pause", help="Pause playback.")
@click.argument("device_id", type=str)
@click.option("--player", "-p", type=str, default=None, help="Target player name.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_pause_cmd(ctx: click.Context, device_id: str, player: Optional[str], json_flag: bool) -> None:
    """Send Pause action to media player."""
    _handle_transport_cmd(ctx, device_id, "Pause", player, json_flag)


@mpris.command(name="play-pause", help="Toggle play/pause state.")
@click.argument("device_id", type=str)
@click.option("--player", "-p", type=str, default=None, help="Target player name.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_play_pause_cmd(ctx: click.Context, device_id: str, player: Optional[str], json_flag: bool) -> None:
    """Toggle PlayPause state on media player."""
    _handle_transport_cmd(ctx, device_id, "PlayPause", player, json_flag)


@mpris.command(name="next", help="Skip to next track.")
@click.argument("device_id", type=str)
@click.option("--player", "-p", type=str, default=None, help="Target player name.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_next_cmd(ctx: click.Context, device_id: str, player: Optional[str], json_flag: bool) -> None:
    """Skip to next track."""
    _handle_transport_cmd(ctx, device_id, "Next", player, json_flag)


@mpris.command(name="previous", help="Skip to previous track.")
@click.argument("device_id", type=str)
@click.option("--player", "-p", type=str, default=None, help="Target player name.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_previous_cmd(ctx: click.Context, device_id: str, player: Optional[str], json_flag: bool) -> None:
    """Skip to previous track."""
    _handle_transport_cmd(ctx, device_id, "Previous", player, json_flag)


@mpris.command(name="stop", help="Stop playback.")
@click.argument("device_id", type=str)
@click.option("--player", "-p", type=str, default=None, help="Target player name.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_stop_cmd(ctx: click.Context, device_id: str, player: Optional[str], json_flag: bool) -> None:
    """Stop playback."""
    _handle_transport_cmd(ctx, device_id, "Stop", player, json_flag)


@mpris.command(name="volume", help="Set player volume percentage (0..100).")
@click.argument("device_id", type=str)
@click.argument("volume", type=int)
@click.option("--player", "-p", type=str, default=None, help="Target player name.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_volume_cmd(ctx: click.Context, device_id: str, volume: int, player: Optional[str], json_flag: bool) -> None:
    """Set volume percentage (0..100) on target player."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    if volume < 0 or volume > 100:
        raise click.BadParameter("Volume must be between 0 and 100", param_hint="VOLUME")

    params: Dict[str, Any] = {"device_id": device_id, "volume": volume}
    if player:
        params["player"] = player.strip()

    res = run_ipc_call(ctx, "mpris_set_volume", params)
    if is_json:
        click.echo(json.dumps(res, indent=2))
        return

    safe_p = escape(res.get("player") or "player")
    console.print(f"[green]✓ Volume set to {volume}% on {safe_p} ({escape(device_id)}).[/]")


@mpris.command(name="seek", help="Seek track position by offset in milliseconds (positive or negative).")
@click.argument("device_id", type=str)
@click.argument("offset_ms", type=int)
@click.option("--player", "-p", type=str, default=None, help="Target player name.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_seek_cmd(ctx: click.Context, device_id: str, offset_ms: int, player: Optional[str], json_flag: bool) -> None:
    """Seek relative track position in milliseconds."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    params: Dict[str, Any] = {"device_id": device_id, "offset_ms": offset_ms}
    if player:
        params["player"] = player.strip()

    res = run_ipc_call(ctx, "mpris_seek", params)
    if is_json:
        click.echo(json.dumps(res, indent=2))
        return

    safe_p = escape(res.get("player") or "player")
    console.print(f"[green]✓ Seek {offset_ms:+d}ms sent to {safe_p} ({escape(device_id)}).[/]")


@mpris.command(name="position", help="Set absolute track position in milliseconds.")
@click.argument("device_id", type=str)
@click.argument("position_ms", type=int)
@click.option("--player", "-p", type=str, default=None, help="Target player name.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def mpris_position_cmd(ctx: click.Context, device_id: str, position_ms: int, player: Optional[str], json_flag: bool) -> None:
    """Set absolute track position in milliseconds."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    if position_ms < 0:
        raise click.BadParameter("Position must be non-negative", param_hint="POSITION_MS")

    params: Dict[str, Any] = {"device_id": device_id, "position_ms": position_ms}
    if player:
        params["player"] = player.strip()

    res = run_ipc_call(ctx, "mpris_set_position", params)
    if is_json:
        click.echo(json.dumps(res, indent=2))
        return

    safe_p = escape(res.get("player") or "player")
    console.print(f"[green]✓ Position set to {position_ms}ms on {safe_p} ({escape(device_id)}).[/]")


# =====================================================================
# Remote Commands CLI (Milestone 15)
# =====================================================================

@cli.group(name="command", help="Discover and trigger commands configured on remote devices.")
def command_group() -> None:
    """Inspect and trigger commands configured on remote paired devices."""
    pass


@command_group.command(name="list", help="List available remote commands on target device.")
@click.argument("device_id", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def command_list_cmd(ctx: click.Context, device_id: str, json_flag: bool) -> None:
    """List available remote commands on target device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip() if device_id else ""
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    params: Dict[str, Any] = {"device_id": device_id}
    res = run_ipc_call(ctx, "list_remote_commands", params)
    if is_json:
        click.echo(json.dumps(res, indent=2))
        return

    commands = res if isinstance(res, list) else []
    if not commands:
        console.print(f"[yellow]No remote commands available for device '[bold]{escape(device_id)}[/]'.[/]")
        return

    table = Table(
        title=f"Remote Commands — {escape(device_id)}",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold white")
    table.add_column("Command", style="dim")

    for cmd in commands:
        k = escape(str(cmd.get("key", "")))
        n = escape(str(cmd.get("name", "")))
        c = str(cmd.get("command", "") or "")
        c_display = escape(c) if c else "[dim]\\[hidden][/]"
        table.add_row(k, n, c_display)

    console.print(table)


@command_group.command(name="exec", help="Trigger execution of a configured remote command.")
@click.argument("device_id", type=str)
@click.argument("key", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def command_exec_cmd(ctx: click.Context, device_id: str, key: str, json_flag: bool) -> None:
    """Trigger execution of a configured remote command on target device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id.strip() if device_id else ""
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")
    key = key.strip() if key else ""
    if not key:
        raise click.UsageError("Argument 'KEY' must not be empty.")

    params: Dict[str, Any] = {"device_id": device_id, "key": key}
    res = run_ipc_call(ctx, "execute_remote_command", params)
    if is_json:
        click.echo(json.dumps(res, indent=2))
        return

    safe_did = escape(device_id)
    safe_key = escape(key)
    panel_text = f"[bold green]✓[/] Execution request for command '[bold cyan]{safe_key}[/]' transmitted to device '[bold]{safe_did}[/]'."
    console.print(Panel(panel_text, title="Remote Command Executed", border_style="green"))


@command_group.command(name="run", help="Alias for 'exec': Trigger execution of a configured remote command.")
@click.argument("device_id", type=str)
@click.argument("key", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def command_run_cmd(ctx: click.Context, device_id: str, key: str, json_flag: bool) -> None:
    """Alias for exec: Trigger execution of a configured remote command on target device."""
    ctx.invoke(command_exec_cmd, device_id=device_id, key=key, json_flag=json_flag)


# =====================================================================
# Volume Control Commands (Milestones 22, 24, 25)
# =====================================================================

def parse_volume_value(val: str | int) -> tuple[int, bool]:
    """Parse volume argument into (integer_volume, is_percent_flag)."""
    s = str(val).strip()
    if s.endswith("%"):
        try:
            pct = max(0, min(100, int(round(float(s[:-1])))))
            return pct, True
        except ValueError:
            raise click.UsageError(f"Invalid volume percentage: '{val}'")
    try:
        iv = int(round(float(s)))
        if 0 <= iv <= 100:
            return iv, True
        return max(0, min(65536, iv)), False
    except ValueError:
        raise click.UsageError(f"Invalid volume value: '{val}'. Expected percentage (0-100) or raw (0-65536).")


def parse_mute_state(val: Optional[str]) -> bool:
    """Parse mute state string ('on'/'off'/'yes'/'no'/'true'/'false'/'1'/'0'). Default True if omitted."""
    if val is None:
        return True
    s = str(val).strip().lower()
    if s in ("on", "true", "1", "yes", "mute"):
        return True
    elif s in ("off", "false", "0", "no", "unmute"):
        return False
    else:
        raise click.UsageError(f"Invalid mute state: '{val}'. Expected 'on' or 'off'.")


@cli.group(name="volume", help="Inspect and control local and remote audio sinks.")
def volume_group() -> None:
    """Audio volume and sink management for local host and remote connected devices."""
    pass


# Local Subgroup
@volume_group.group(name="local", help="Manage local host audio output sinks.")
def volume_local_group() -> None:
    """Inspect and adjust local host sound output sinks."""
    pass


@volume_local_group.command(name="list", help="List available local audio output sinks.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_local_list_cmd(ctx: click.Context, json_flag: bool) -> None:
    """List available local audio output sinks."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    res = run_ipc_call(ctx, "systemvolume_get_sinks")
    sinks = res if isinstance(res, list) else []

    if is_json:
        click.echo(json.dumps(sinks, indent=2))
        return

    if not sinks:
        console.print("[yellow]No local audio sinks found.[/]")
        return

    table = Table(
        title="Local Audio Sinks",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="bold white")
    table.add_column("Volume %", justify="right")
    table.add_column("Muted", justify="center")
    table.add_column("Default", justify="center")

    for s in sinks:
        name = escape(str(s.get("name", "")))
        desc = escape(str(s.get("description", "") or name))
        vol_pct = s.get("volume_percent", 0)
        raw_vol = s.get("volume", 0)
        vol_display = f"{vol_pct}% [dim]({raw_vol})[/]"
        muted = "[bold red]Yes[/]" if s.get("muted") else "[green]No[/]"
        is_default = "[bold green]● Default[/]" if s.get("enabled") else "[dim]-[/]"
        table.add_row(name, desc, vol_display, muted, is_default)

    console.print(table)


@volume_local_group.command(name="set", help="Set volume percentage or raw level on a local sink.")
@click.argument("sink", type=str)
@click.argument("volume", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_local_set_cmd(ctx: click.Context, sink: str, volume: str, json_flag: bool) -> None:
    """Set volume level on local audio output sink."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    sink = sink.strip()
    if not sink:
        raise click.UsageError("Argument 'SINK' must not be empty.")

    vol_val, is_pct = parse_volume_value(volume)
    raw_vol = round((vol_val / 100.0) * 65536) if is_pct else vol_val
    params = {"name": sink, "volume": raw_vol}
    res = run_ipc_call(ctx, "systemvolume_set_volume", params)

    if is_json:
        out = {
            "sink": sink,
            "volume": raw_vol,
            "volume_percent": vol_val if is_pct else round((raw_vol / 65536.0) * 100),
            "success": res.get("success", True),
        }
        click.echo(json.dumps(out, indent=2))
        return

    vol_disp = f"{vol_val}%" if is_pct else str(vol_val)
    console.print(f"[bold green]✓[/] Volume of sink '[bold]{escape(sink)}[/]' set to {vol_disp}.")


@volume_local_group.command(name="mute", help="Set mute state on local audio sink.")
@click.argument("sink", type=str)
@click.argument("state", type=str, required=False, default=None)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_local_mute_cmd(ctx: click.Context, sink: str, state: Optional[str], json_flag: bool) -> None:
    """Set mute state on local audio output sink."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    sink = sink.strip()
    if not sink:
        raise click.UsageError("Argument 'SINK' must not be empty.")

    muted = parse_mute_state(state)
    params = {"name": sink, "muted": muted}
    res = run_ipc_call(ctx, "systemvolume_set_muted", params)

    if is_json:
        out = {"sink": sink, "muted": muted, "success": res.get("success", True)}
        click.echo(json.dumps(out, indent=2))
        return

    st_str = "[bold red]muted[/]" if muted else "[bold green]unmuted[/]"
    console.print(f"[bold green]✓[/] Local sink '[bold]{escape(sink)}[/]' is now {st_str}.")


@volume_local_group.command(name="default", help="Set default local audio output sink.")
@click.argument("sink", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_local_default_cmd(ctx: click.Context, sink: str, json_flag: bool) -> None:
    """Set default local audio output sink."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    sink = sink.strip()
    if not sink:
        raise click.UsageError("Argument 'SINK' must not be empty.")

    params = {"name": sink}
    res = run_ipc_call(ctx, "systemvolume_set_default", params)

    if is_json:
        out = {"sink": sink, "default": True, "success": res.get("success", True)}
        click.echo(json.dumps(out, indent=2))
        return

    console.print(f"[bold green]✓[/] Default local sink set to '[bold]{escape(sink)}[/]'.")


# Convenience Top-Level Shortcuts on volume
@volume_group.command(name="list", help="Shortcut: List available local audio sinks.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_top_list_cmd(ctx: click.Context, json_flag: bool) -> None:
    """Shortcut: List available local audio output sinks."""
    ctx.invoke(volume_local_list_cmd, json_flag=json_flag)


@volume_group.command(name="set", help="Shortcut: Set volume on local sink.")
@click.argument("sink", type=str)
@click.argument("volume", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_top_set_cmd(ctx: click.Context, sink: str, volume: str, json_flag: bool) -> None:
    """Shortcut: Set volume level on local audio output sink."""
    ctx.invoke(volume_local_set_cmd, sink=sink, volume=volume, json_flag=json_flag)


@volume_group.command(name="mute", help="Shortcut: Set mute state on local sink.")
@click.argument("sink", type=str)
@click.argument("state", type=str, required=False, default=None)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_top_mute_cmd(ctx: click.Context, sink: str, state: Optional[str], json_flag: bool) -> None:
    """Shortcut: Set mute state on local audio output sink."""
    ctx.invoke(volume_local_mute_cmd, sink=sink, state=state, json_flag=json_flag)


# Remote Subgroup with Flexible Device ID Handling
class RemoteVolumeGroup(click.Group):
    """Click group for remote volume commands allowing syntax:
    'pengulink volume remote <device_id> <subcommand>' OR
    'pengulink volume remote <subcommand> <device_id>'
    """

    def parse_args(self, ctx: click.Context, args: list) -> list:
        if args and args[0] not in self.commands and args[0] not in ("--help", "-h"):
            device_id = args.pop(0)
            ctx.ensure_object(dict)["remote_device_id"] = device_id
            if not args or (args and args[0] not in self.commands and args[0] not in ("--help", "-h")):
                args = ["list"] + list(args)
        return super().parse_args(ctx, args)


@volume_group.group(
    name="remote",
    cls=RemoteVolumeGroup,
    help="Inspect and control audio sinks on remote connected devices.",
)
def volume_remote_group() -> None:
    """Inspect and adjust audio output sinks on remote paired devices."""
    pass


@volume_remote_group.command(name="list", help="List audio sinks on remote device.")
@click.argument("device_id_arg", type=str, required=False, default=None)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_remote_list_cmd(ctx: click.Context, device_id_arg: Optional[str], json_flag: bool) -> None:
    """List audio sinks reported by remote device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = (device_id_arg or (ctx.obj.get("remote_device_id") if ctx.obj else None) or "").strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    params = {"device_id": device_id}
    res = run_ipc_call(ctx, "systemvolume_get_remote_sinks", params)
    sinks = res if isinstance(res, list) else []

    if is_json:
        click.echo(json.dumps(sinks, indent=2))
        return

    if not sinks:
        console.print(f"[yellow]No remote audio sinks cached for device '[bold]{escape(device_id)}[/]'.[/]")
        return

    table = Table(
        title=f"Remote Audio Sinks — {escape(device_id)}",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="bold white")
    table.add_column("Volume %", justify="right")
    table.add_column("Muted", justify="center")
    table.add_column("Default", justify="center")

    for s in sinks:
        name = escape(str(s.get("name", "")))
        desc = escape(str(s.get("description", "") or name))
        vol_pct = s.get("volume_percent", 0)
        raw_vol = s.get("volume", 0)
        vol_display = f"{vol_pct}% [dim]({raw_vol})[/]"
        muted = "[bold red]Yes[/]" if s.get("muted") else "[green]No[/]"
        is_default = "[bold green]● Default[/]" if s.get("enabled") else "[dim]-[/]"
        table.add_row(name, desc, vol_display, muted, is_default)

    console.print(table)


@volume_remote_group.command(name="set", help="Set volume on remote device audio sink.")
@click.argument("arg1", type=str)
@click.argument("arg2", type=str)
@click.argument("arg3", type=str, required=False, default=None)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_remote_set_cmd(
    ctx: click.Context, arg1: str, arg2: str, arg3: Optional[str], json_flag: bool
) -> None:
    """Set volume level on remote device audio sink."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    if arg3 is not None:
        device_id, sink, volume_str = arg1, arg2, arg3
    else:
        device_id = (ctx.obj.get("remote_device_id") if ctx.obj else None) or ""
        sink, volume_str = arg1, arg2

    device_id = device_id.strip()
    sink = sink.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")
    if not sink:
        raise click.UsageError("Argument 'SINK' must not be empty.")

    vol_val, is_pct = parse_volume_value(volume_str)
    raw_vol = round((vol_val / 100.0) * 65536) if is_pct else vol_val
    params = {"device_id": device_id, "name": sink, "volume": raw_vol}
    res = run_ipc_call(ctx, "systemvolume_set_remote_volume", params)

    if is_json:
        out = {
            "device_id": device_id,
            "sink": sink,
            "volume": raw_vol,
            "volume_percent": vol_val if is_pct else round((raw_vol / 65536.0) * 100),
            "success": res.get("success", True),
        }
        click.echo(json.dumps(out, indent=2))
        return

    vol_disp = f"{vol_val}%" if is_pct else str(raw_vol)
    console.print(f"[bold green]✓[/] Remote sink '[bold]{escape(sink)}[/]' volume set to {vol_disp} on device '[bold]{escape(device_id)}[/]'.")


@volume_remote_group.command(name="mute", help="Set mute state on remote device audio sink.")
@click.argument("arg1", type=str)
@click.argument("arg2", type=str, required=False, default=None)
@click.argument("arg3", type=str, required=False, default=None)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_remote_mute_cmd(
    ctx: click.Context, arg1: str, arg2: Optional[str], arg3: Optional[str], json_flag: bool
) -> None:
    """Set mute state on remote device audio sink."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    saved_did = (ctx.obj.get("remote_device_id") if ctx.obj else None) or ""
    if arg3 is not None:
        device_id, sink, state_str = arg1, arg2, arg3
    elif saved_did:
        device_id = saved_did
        sink = arg1
        state_str = arg2
    else:
        device_id = arg1
        sink = arg2 or ""
        state_str = None

    device_id = device_id.strip()
    sink = (sink or "").strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")
    if not sink:
        raise click.UsageError("Argument 'SINK' must not be empty.")

    muted = parse_mute_state(state_str)
    params = {"device_id": device_id, "name": sink, "muted": muted}
    res = run_ipc_call(ctx, "systemvolume_set_remote_muted", params)

    if is_json:
        out = {"device_id": device_id, "sink": sink, "muted": muted, "success": res.get("success", True)}
        click.echo(json.dumps(out, indent=2))
        return

    st_str = "[bold red]muted[/]" if muted else "[bold green]unmuted[/]"
    console.print(f"[bold green]✓[/] Remote sink '[bold]{escape(sink)}[/]' is now {st_str} on device '[bold]{escape(device_id)}[/]'.")


@volume_remote_group.command(name="default", help="Set default audio sink on remote device.")
@click.argument("arg1", type=str)
@click.argument("arg2", type=str, required=False, default=None)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_remote_default_cmd(ctx: click.Context, arg1: str, arg2: Optional[str], json_flag: bool) -> None:
    """Set default audio output sink on remote device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    if arg2 is not None:
        device_id, sink = arg1, arg2
    else:
        device_id = (ctx.obj.get("remote_device_id") if ctx.obj else None) or ""
        sink = arg1

    device_id = device_id.strip()
    sink = sink.strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")
    if not sink:
        raise click.UsageError("Argument 'SINK' must not be empty.")

    params = {"device_id": device_id, "name": sink}
    res = run_ipc_call(ctx, "systemvolume_set_remote_default", params)

    if is_json:
        out = {"device_id": device_id, "sink": sink, "default": True, "success": res.get("success", True)}
        click.echo(json.dumps(out, indent=2))
        return

    console.print(f"[bold green]✓[/] Default remote sink set to '[bold]{escape(sink)}[/]' on device '[bold]{escape(device_id)}[/]'.")


@volume_remote_group.command(name="refresh", help="Request fresh audio sink list from remote device.")
@click.argument("device_id_arg", type=str, required=False, default=None)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def volume_remote_refresh_cmd(ctx: click.Context, device_id_arg: Optional[str], json_flag: bool) -> None:
    """Request remote peer to re-advertise its complete audio sink list."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = (device_id_arg or (ctx.obj.get("remote_device_id") if ctx.obj else None) or "").strip()
    if not device_id:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    params = {"device_id": device_id}
    res = run_ipc_call(ctx, "systemvolume_request_remote_sinks", params)

    if is_json:
        out = {"device_id": device_id, "success": res.get("success", True), "message": "Remote sink list requested"}
        click.echo(json.dumps(out, indent=2))
        return

    console.print(f"[bold green]✓[/] Requested audio sink list update from remote device '[bold]{escape(device_id)}[/]'.")


# =====================================================================
# RunCommand Management Commands (Milestones 23, 25)
# =====================================================================

@cli.group(name="runcommand", help="Manage and trigger locally configured host commands.")
def runcommand_group() -> None:
    """Configure and execute local commands exposed to paired KDE Connect peers."""
    pass


@runcommand_group.command(name="list", help="List all locally configured commands.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def runcommand_list_cmd(ctx: click.Context, json_flag: bool) -> None:
    """List locally configured commands."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    res = run_ipc_call(ctx, "runcommand_list")
    commands = res if isinstance(res, list) else []

    if is_json:
        click.echo(json.dumps(commands, indent=2))
        return

    if not commands:
        console.print("[yellow]No local commands configured.[/]")
        return

    table = Table(
        title="Local Commands",
        box=box.ROUNDED,
        header_style="bold green",
    )
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold white")
    table.add_column("Command", style="dim")
    table.add_column("Enabled", justify="center")

    for cmd in commands:
        k = escape(str(cmd.get("key", "")))
        n = escape(str(cmd.get("name", "")))
        c = escape(str(cmd.get("command", "") or ""))
        en = "[bold green]Yes[/]" if cmd.get("enabled", True) else "[red]No[/]"
        table.add_row(k, n, c, en)

    console.print(table)


@runcommand_group.command(name="add", help="Add a locally configured command entry.")
@click.argument("name", type=str)
@click.argument("command", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def runcommand_add_cmd(ctx: click.Context, name: str, command: str, json_flag: bool) -> None:
    """Add a new command to local command store."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    name = name.strip()
    command = command.strip()
    if not name:
        raise click.UsageError("Argument 'NAME' must not be empty.")
    if not command:
        raise click.UsageError("Argument 'COMMAND' must not be empty.")

    params = {"name": name, "command": command}
    res = run_ipc_call(ctx, "runcommand_add", params)

    if is_json:
        click.echo(json.dumps(res, indent=2))
        return

    key = res.get("key", "")
    console.print(f"[bold green]✓[/] Added command '[bold]{escape(name)}[/]' with key '[bold cyan]{escape(key)}[/]'.")


@runcommand_group.command(name="remove", help="Remove a locally configured command entry by key.")
@click.argument("key", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def runcommand_remove_cmd(ctx: click.Context, key: str, json_flag: bool) -> None:
    """Remove a command from local command store by key."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    key = key.strip()
    if not key:
        raise click.UsageError("Argument 'KEY' must not be empty.")

    params = {"key": key}
    res = run_ipc_call(ctx, "runcommand_remove", params)

    if is_json:
        out = {"key": key, "success": res.get("success", False)}
        click.echo(json.dumps(out, indent=2))
        return

    if res.get("success"):
        console.print(f"[bold green]✓[/] Removed command '[bold cyan]{escape(key)}[/]'.")
    else:
        console.print(f"[bold red]✕[/] Command '[bold cyan]{escape(key)}[/]' not found.")


@runcommand_group.command(name="run", help="Execute a locally configured command by key.")
@click.argument("key", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Output machine-readable JSON.")
@click.pass_context
def runcommand_run_cmd(ctx: click.Context, key: str, json_flag: bool) -> None:
    """Execute a configured local command and print output."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    key = key.strip()
    if not key:
        raise click.UsageError("Argument 'KEY' must not be empty.")

    params = {"key": key}
    res = run_ipc_call(ctx, "runcommand_exec", params)

    if is_json:
        click.echo(json.dumps(res, indent=2))
        return

    success = bool(res.get("success"))
    code = res.get("exit_code", 0)
    stdout_lines = res.get("stdout", [])
    stderr_lines = res.get("stderr", [])

    if success:
        console.print(f"[bold green]✓ Command succeeded (exit code {code})[/]")
    else:
        console.print(f"[bold red]✕ Command failed (exit code {code})[/]")

    if stdout_lines:
        console.print("[bold]Standard Output:[/]")
        for line in stdout_lines:
            console.print(f"  {escape(line.rstrip())}")

    if stderr_lines:
        console.print("[bold red]Standard Error:[/]")
        for line in stderr_lines:
            console.print(f"  [red]{escape(line.rstrip())}[/]")


# =====================================================================
# Telephony Commands (Milestone 29)
# =====================================================================

@cli.group(name="telephony", help="Manage and inspect telephony call states and actions.")
def telephony_group() -> None:
    """Manage and inspect telephony call states and actions."""


@telephony_group.command(name="calls", help="Display active phone calls.")
@click.argument("device_id_arg", required=False, default=None, metavar="[DEVICE_ID]")
@click.option(
    "--device-id",
    "-d",
    "device_id_opt",
    default=None,
    help="Filter active calls by remote device ID.",
)
@click.option(
    "--json",
    "json_flag",
    is_flag=True,
    default=False,
    help="Output machine-readable JSON.",
)
@click.pass_context
def telephony_calls_cmd(
    ctx: click.Context,
    device_id_arg: str | None,
    device_id_opt: str | None,
    json_flag: bool,
) -> None:
    """Display active phone calls across devices or filtered by device ID."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    device_id = device_id_opt or device_id_arg
    if device_id:
        device_id = device_id.strip()

    params: dict[str, Any] = {}
    if device_id:
        params["device_id"] = device_id

    res = run_ipc_call(ctx, "telephony_get_calls", params)
    calls = res if isinstance(res, list) else []

    if is_json:
        click.echo(json.dumps(calls, indent=2))
        return

    if not calls:
        if device_id:
            console.print(f"[yellow]No active calls for device '[bold cyan]{escape(device_id)}[/]'.[/]")
        else:
            console.print("[yellow]No active phone calls.[/]")
        return

    title = f"Active Phone Calls ({escape(device_id)})" if device_id else "Active Phone Calls"
    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Device ID", style="cyan", no_wrap=True)
    table.add_column("Phone Number", style="bold white")
    table.add_column("Contact Name", style="bold green")
    table.add_column("State", justify="center")
    table.add_column("Duration", justify="right")

    for call in calls:
        d_id = escape(str(call.get("device_id", "")))
        num = escape(str(call.get("phone_number", "") or "Private/Unknown"))
        name = escape(str(call.get("contact_name", "") or "Unknown"))
        st = format_call_state_badge(str(call.get("state", "")))
        dur = format_call_duration(call.get("duration"))
        table.add_row(d_id, num, name, st, dur)

    console.print(table)


@telephony_group.command(name="mute-ringer", help="Request remote device to mute its incoming call ringer.")
@click.argument("device_id", type=str)
@click.option(
    "--json",
    "json_flag",
    is_flag=True,
    default=False,
    help="Output machine-readable JSON.",
)
@click.pass_context
def telephony_mute_ringer_cmd(ctx: click.Context, device_id: str, json_flag: bool) -> None:
    """Send request to mute incoming call ringer on remote device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    clean_did = device_id.strip() if device_id else ""
    if not clean_did:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    params = {"device_id": clean_did}
    res = run_ipc_call(ctx, "telephony_mute_ringer", params)

    if is_json:
        click.echo(json.dumps(res, indent=2))
        return

    console.print(f"[bold green]✓[/] Requested ringer mute for device '[bold cyan]{escape(clean_did)}[/]'.")


@telephony_group.command(name="reject-call", help="Request remote device to reject the active or incoming call.")
@click.argument("device_id", type=str)
@click.option("--phone-number", type=str, default=None, help="Optional phone number to decline.")
@click.option(
    "--json",
    "json_flag",
    is_flag=True,
    default=False,
    help="Output machine-readable JSON.",
)
@click.pass_context
def telephony_reject_call_cmd(
    ctx: click.Context,
    device_id: str,
    phone_number: str | None,
    json_flag: bool,
) -> None:
    """Send request to reject active/incoming call on remote device."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    clean_did = device_id.strip() if device_id else ""
    if not clean_did:
        raise click.UsageError("Argument 'DEVICE_ID' must not be empty.")

    params: dict[str, Any] = {"device_id": clean_did}
    if phone_number:
        params["phone_number"] = phone_number.strip()
    res = run_ipc_call(ctx, "telephony_reject_call", params)

    if is_json:
        click.echo(json.dumps(res, indent=2))
        return

    console.print(f"[bold green]✓[/] Requested call rejection for device '[bold cyan]{escape(clean_did)}[/]'.")


# =====================================================================
# PauseMusic Commands (Milestone 29)
# =====================================================================

@cli.group(name="pausemusic", help="Inspect and configure media pause policies on phone calls.")
def pausemusic_group() -> None:
    """Manage automatic media pause and audio mute behavior during calls."""


@pausemusic_group.command(name="status", help="Display current PauseMusic configuration and interruption state.")
@click.option(
    "--json",
    "json_flag",
    is_flag=True,
    default=False,
    help="Output machine-readable JSON.",
)
@click.pass_context
def pausemusic_status_cmd(ctx: click.Context, json_flag: bool) -> None:
    """Display PauseMusic configuration and live media interruption state."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))
    status_info = run_ipc_call(ctx, "pausemusic_get_config", {})

    cfg = status_info.get("config", {}) if isinstance(status_info, dict) else {}
    output_dict: dict[str, Any] = dict(status_info) if isinstance(status_info, dict) else {}
    if isinstance(cfg, dict):
        output_dict.update(cfg)

    if is_json:
        click.echo(json.dumps(output_dict, indent=2))
        return

    cond = output_dict.get("condition", "ringing")
    cond_talking = output_dict.get("condition_talking", cond == "talking")
    cond_str = "[cyan]talking[/] (pause after call is answered)" if cond_talking else "[cyan]ringing[/] (pause immediately on incoming ring)"

    act = output_dict.get("action", "pause")
    action_pause = output_dict.get("action_pause", act in ("pause", "both"))
    action_mute = output_dict.get("action_mute", act in ("mute", "both"))
    action_resume = output_dict.get("action_resume", output_dict.get("resume", True))

    pause_badge = "[bold green]Enabled[/]" if action_pause else "[dim]Disabled[/]"
    mute_badge = "[bold green]Enabled[/]" if action_mute else "[dim]Disabled[/]"
    resume_badge = "[bold green]Enabled[/]" if action_resume else "[dim]Disabled[/]"

    is_interrupted = output_dict.get("is_interrupted", False)
    interrupted_badge = "[bold yellow]● ACTIVE INTERRUPTION[/]" if is_interrupted else "[bold green]○ Idle / Normal[/]"

    active_devs = output_dict.get("active_call_devices", [])
    devs_str = ", ".join(f"[cyan]{escape(d)}[/]" for d in active_devs) if active_devs else "[dim]None[/]"

    paused_plyrs = output_dict.get("paused_players", [])
    plyrs_str = ", ".join(f"[magenta]{escape(p)}[/]" for p in paused_plyrs) if paused_plyrs else "[dim]None[/]"

    muted_snks = output_dict.get("muted_sinks", [])
    snks_str = ", ".join(f"[yellow]{escape(s)}[/]" for s in muted_snks) if muted_snks else "[dim]None[/]"

    body = (
        f"[bold]Trigger Condition:[/]  {cond_str}\n"
        f"[bold]Action Policy:[/]      [bold cyan]{act}[/] (MPRIS: {pause_badge}, Audio Sinks: {mute_badge})\n"
        f"[bold]Auto-Resume:[/]        {resume_badge}\n"
        f"[dim]──────────────────────────────────────────────[/]\n"
        f"[bold]Interruption Status:[/] {interrupted_badge}\n"
        f"[bold]Active Devices:[/]      {devs_str}\n"
        f"[bold]Paused Players:[/]      {plyrs_str}\n"
        f"[bold]Muted Sinks:[/]         {snks_str}"
    )

    panel = Panel(body, title="[bold cyan]PauseMusic Status & Policy[/]", box=box.ROUNDED, expand=False)
    console.print(panel)


@pausemusic_group.command(name="configure", help="Configure automatic pause/mute policies for calls.")
@click.option(
    "--condition-talking",
    "--condition",
    "condition",
    type=click.Choice(["ringing", "talking"], case_sensitive=False),
    default=None,
    help="Trigger condition: 'ringing' (pause on ring) or 'talking' (pause on answer).",
)
@click.option(
    "--action",
    type=click.Choice(["pause", "mute", "both", "none"], case_sensitive=False),
    default=None,
    help="Interruption action: 'pause', 'mute', 'both', or 'none'.",
)
@click.option(
    "--action-pause/--no-action-pause",
    "action_pause",
    default=None,
    help="Enable or disable MPRIS player pausing.",
)
@click.option(
    "--action-mute/--no-action-mute",
    "action_mute",
    default=None,
    help="Enable or disable audio output sink muting.",
)
@click.option(
    "--action-resume/--no-action-resume",
    "--resume/--no-resume",
    "action_resume",
    default=None,
    help="Enable or disable automatic media/audio resumption when call ends.",
)
@click.option(
    "--json",
    "json_flag",
    is_flag=True,
    default=False,
    help="Output machine-readable JSON.",
)
@click.pass_context
def pausemusic_configure_cmd(
    ctx: click.Context,
    condition: str | None,
    action: str | None,
    action_pause: bool | None,
    action_mute: bool | None,
    action_resume: bool | None,
    json_flag: bool,
) -> None:
    """Update PauseMusic policy configuration."""
    is_json = json_flag or (bool(ctx.obj and ctx.obj.get("json_output")))

    if (
        condition is None
        and action is None
        and action_pause is None
        and action_mute is None
        and action_resume is None
    ):
        raise click.UsageError(
            "At least one configuration option must be specified (e.g. '--condition', '--action', '--action-pause')."
        )

    params: dict[str, Any] = {}
    if condition is not None:
        params["condition"] = condition.lower()
    if action is not None:
        params["action"] = action.lower()
    if action_pause is not None:
        params["action_pause"] = action_pause
    if action_mute is not None:
        params["action_mute"] = action_mute
    if action_resume is not None:
        params["action_resume"] = action_resume
        params["resume"] = action_resume

    res = run_ipc_call(ctx, "pausemusic_set_config", params)

    cfg = res.get("config", {}) if isinstance(res, dict) else {}
    output_dict: dict[str, Any] = dict(res) if isinstance(res, dict) else {}
    if isinstance(cfg, dict):
        output_dict.update(cfg)

    if is_json:
        click.echo(json.dumps(output_dict, indent=2))
        return

    console.print("[bold green]✓[/] Updated PauseMusic configuration:")
    console.print(f"  [bold]Condition:[/] [cyan]{output_dict.get('condition')}[/]")
    console.print(
        f"  [bold]Action:[/]    [cyan]{output_dict.get('action')}[/] (pause={output_dict.get('action_pause')}, mute={output_dict.get('action_mute')})"
    )
    console.print(f"  [bold]Resume:[/]    [cyan]{output_dict.get('action_resume', output_dict.get('resume'))}[/]")


if __name__ == "__main__":
    cli()
