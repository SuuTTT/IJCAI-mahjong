import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SENTINEL = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
DEFAULT_RETRY_SECONDS = 5
ACTION_PATTERN = re.compile(
    r"^(PASS|HU|PLAY(?:\s+\S+)?|GANG(?:\s+\S+)?|BUGANG(?:\s+\S+)?|PENG(?:\s+\S+\s+\S+)?|CHI(?:\s+\S+\s+\S+)?)$"
)


def _stderr_tail(proc: subprocess.Popen, max_chars: int = 4000) -> str:
    if proc.stderr is None:
        return ""
    try:
        if proc.poll() is None:
            return ""
        text = proc.stderr.read() or ""
        text = text.strip()
        if len(text) > max_chars:
            return text[-max_chars:]
        return text
    except Exception:
        return ""


class BotProcess:
    def __init__(self, bot_cmd: Sequence[str], bot_cwd: Optional[Path]):
        self._bot_cmd = list(bot_cmd)
        self._bot_cwd = str(bot_cwd) if bot_cwd else None
        self._proc: Optional[subprocess.Popen] = None
        self._start()

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            self._bot_cmd,
            cwd=self._bot_cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._write_line("1")
        time.sleep(1.0)
        rc = self._proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"Bot exited during startup handshake (code={rc}). stderr={_stderr_tail(self._proc) or '<empty>'}"
            )

    def _write_line(self, line: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Bot stdin is unavailable.")
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:
            rc = self._proc.poll() if self._proc else None
            tail = _stderr_tail(self._proc) if self._proc else ""
            raise RuntimeError(
                f"Write to bot stdin failed: {exc}; code={rc}; stderr={tail or '<empty>'}"
            ) from exc

    def _read_action(self) -> str:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Bot stdout is unavailable.")

        lines: List[str] = []
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                rc = self._proc.poll()
                if rc is None:
                    time.sleep(0.15)
                    rc = self._proc.poll()
                raise RuntimeError(
                    f"Bot stdout closed unexpectedly (code={rc}). stderr={_stderr_tail(self._proc) or '<empty>'}"
                )
            text = line.rstrip("\r\n")
            if text == SENTINEL:
                break
            if text:
                lines.append(text)

        for item in lines:
            if ACTION_PATTERN.match(item):
                return item
        return lines[0] if lines else "PASS"

    def ask(self, command: str) -> str:
        if not command.strip():
            return "PASS"
        self._write_line(command)
        response = self._read_action().strip()
        return response or "PASS"

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None


@dataclass
class MatchState:
    bot: BotProcess
    pending_response: str = ""
    last_command: str = ""
    last_response: str = "PASS"


def _parse_head(line: str) -> Tuple[int, int]:
    parts = line.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid localai head line: {line!r}")
    return int(parts[0]), int(parts[1])


def _extract_commands(request_text: str) -> List[str]:
    text = request_text.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [text]

    if isinstance(payload, str):
        return [payload.strip()]
    if isinstance(payload, dict) and isinstance(payload.get("request"), str):
        return [payload["request"].strip()]
    return [text]


def _handle_requests(
    matches: Dict[str, MatchState],
    request_pairs: List[Tuple[str, str]],
    bot_cmd: Sequence[str],
    bot_cwd: Optional[Path],
) -> None:
    for match_id, request_text in request_pairs:
        state = matches.get(match_id)
        if state is None:
            state = MatchState(bot=BotProcess(bot_cmd=bot_cmd, bot_cwd=bot_cwd))
            matches[match_id] = state

        commands = _extract_commands(request_text)
        if not commands:
            continue

        final_response = state.last_response
        for command in commands:
            if not command:
                continue
            if command == state.last_command:
                final_response = state.last_response
                continue
            try:
                final_response = state.bot.ask(command)
            except Exception as exc:
                preview = command if len(command) <= 200 else command[:200] + "..."
                print(
                    f"[localai][{match_id}] bot execution failed. command={preview!r}. error={exc}",
                    file=sys.stderr,
                )
                final_response = "PASS"
            state.last_command = command
            state.last_response = final_response

        state.pending_response = final_response


def _handle_finished(matches: Dict[str, MatchState], finished_lines: List[str]) -> None:
    for line in finished_lines:
        line = line.strip()
        if not line:
            continue
        match_id = line.split(maxsplit=1)[0]
        state = matches.pop(match_id, None)
        if state is not None:
            state.bot.close()


def _poll_once(
    localai_url: str,
    matches: Dict[str, MatchState],
    bot_cmd: Sequence[str],
    bot_cwd: Optional[Path],
) -> None:
    req = urllib.request.Request(localai_url)
    sent_match_ids: List[str] = []
    for match_id, state in matches.items():
        if state.pending_response:
            req.add_header(f"X-Match-{match_id}", state.pending_response)
            sent_match_ids.append(match_id)

    with urllib.request.urlopen(req, timeout=None) as resp:
        payload = resp.read().decode("utf-8")

    for match_id in sent_match_ids:
        if match_id in matches:
            matches[match_id].pending_response = ""

    lines = payload.splitlines()
    if not lines:
        return

    m, n = _parse_head(lines[0])
    request_end = 1 + 2 * m
    request_lines = lines[1:request_end]
    result_lines = lines[request_end : request_end + n]

    request_pairs: List[Tuple[str, str]] = []
    for i in range(0, len(request_lines), 2):
        if i + 1 >= len(request_lines):
            break
        match_id = request_lines[i].strip()
        request_text = request_lines[i + 1].strip()
        if match_id and request_text:
            request_pairs.append((match_id, request_text))

    _handle_requests(matches, request_pairs, bot_cmd=bot_cmd, bot_cwd=bot_cwd)
    _handle_finished(matches, result_lines)


def _cleanup(matches: Dict[str, MatchState]) -> None:
    for state in matches.values():
        state.bot.close()
    matches.clear()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple localai adapter for black-box Mahjong bot."
    )
    parser.add_argument(
        "--localai-url",
        required=True,
        help="Full localai endpoint URL.",
    )
    parser.add_argument(
        "--bot-cmd",
        nargs="+",
        required=True,
        help="Bot command, e.g. python local_bots\\mahjong\\llm_bot.py",
    )
    parser.add_argument(
        "--bot-cwd",
        help="Working directory for the bot process.",
    )
    parser.add_argument(
        "--retry-seconds",
        type=int,
        default=DEFAULT_RETRY_SECONDS,
        help="Retry interval on errors.",
    )
    args = parser.parse_args()

    bot_cmd = list(args.bot_cmd)
    bot_cwd = Path(args.bot_cwd).resolve() if args.bot_cwd else None
    retry_seconds = max(args.retry_seconds, 1)

    try:
        probe = BotProcess(bot_cmd=bot_cmd, bot_cwd=bot_cwd)
        probe.close()
    except Exception as exc:
        raise RuntimeError(
            f"Bot launch validation failed. cmd={bot_cmd}, cwd={bot_cwd}. {exc}"
        )

    matches: Dict[str, MatchState] = {}
    try:
        while True:
            try:
                _poll_once(
                    localai_url=args.localai_url,
                    matches=matches,
                    bot_cmd=bot_cmd,
                    bot_cwd=bot_cwd,
                )
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                print(f"[localai] retry after transport/parse error: {exc}", file=sys.stderr)
                time.sleep(retry_seconds)
            except Exception as exc:
                print(f"[localai] retry after runtime error: {exc}", file=sys.stderr)
                time.sleep(retry_seconds)
    finally:
        _cleanup(matches)


if __name__ == "__main__":
    main()
