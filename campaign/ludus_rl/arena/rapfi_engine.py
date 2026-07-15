"""Rapfi wrapper — strong Gomoku opponent via the Gomocup pbrain protocol.

Rapfi (github.com/dhbloo/rapfi, GPL-3) runs as an EXTERNAL PROCESS; nothing
links against it and it ships separately from this MIT codebase. One engine
process is kept per live game and fed incremental TURN commands.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

RAPFI_BIN = Path("/root/rapfi/Rapfi/build/pbrain-rapfi")
BOARD = 15
_LOCK = threading.Lock()


def available() -> bool:
    return RAPFI_BIN.exists()


class RapfiGame:
    """One engine process per match. Coordinates are (x, y) = (col, row)."""

    def __init__(self, timeout_ms: int = 1500):
        self.p = subprocess.Popen(
            [str(RAPFI_BIN)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            cwd=str(RAPFI_BIN.parent))
        self._cmd(f"START {BOARD}", expect_ok=True)
        self._cmd(f"INFO timeout_turn {timeout_ms}")
        self._cmd("INFO rule 0")            # 0 = freestyle (five or more wins)
        self._cmd("INFO thread_num 4")

    def _cmd(self, line: str, expect_ok: bool = False) -> str | None:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        if expect_ok:
            while True:
                out = self.p.stdout.readline().strip()
                if not out or out.startswith(("OK", "ERROR")):
                    return out
        return None

    def _read_move(self) -> tuple[int, int]:
        while True:
            out = self.p.stdout.readline().strip()
            if not out:
                raise RuntimeError("rapfi died")
            if out.startswith(("MESSAGE", "DEBUG", "OK")):
                continue
            x, y = out.split(",")
            return int(x), int(y)

    def opponent_move_and_reply(self, x: int, y: int) -> tuple[int, int]:
        """Tell rapfi the human played (x, y); returns rapfi's reply move."""
        self.p.stdin.write(f"TURN {x},{y}\n")
        self.p.stdin.flush()
        return self._read_move()

    def begin(self) -> tuple[int, int]:
        """Rapfi opens the game (when it plays black)."""
        self.p.stdin.write("BEGIN\n")
        self.p.stdin.flush()
        return self._read_move()

    def close(self):
        try:
            self.p.stdin.write("END\n")
            self.p.stdin.flush()
            self.p.terminate()
        except Exception:
            pass
