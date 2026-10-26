"""Copies everything a CLI invocation prints (stdout and stderr) into <run_dir>\\logs\\spore_xl_<stage-or-all>_<YYYYMMDD_HHMMSS>.log.

The tee is installed first thing, so text printed before the run directory is known is held and written when `attach` is called.
Worker processes of a --workers pool cannot share the tee; they append their logger lines to the same file (see xl_common.Run.logger)
through the XL_CONSOLE_LOG / XL_CONSOLE_PARENT_PID environment variables set here.
"""
import os
import sys
import time
from pathlib import Path


class _Tee:
    def __init__(self, owner, stream):
        self.owner, self.stream = owner, stream

    def write(self, text):
        self.stream.write(text)
        self.owner.record(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        if self.owner.handle:
            self.owner.handle.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno ...
        return getattr(self.stream, name)


class ConsoleLog:
    def __init__(self):
        self.handle = None
        self.pending = []
        self.path = None
        self._original = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = _Tee(self, sys.stdout), _Tee(self, sys.stderr)

    def record(self, text):
        if self.handle:
            self.handle.write(text)
        else:
            self.pending.append(text)

    def attach(self, log_dir, label):
        if self.handle:
            return self.path
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"spore_xl_{label}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        counter = 1
        while self.path.exists():  # two invocations within one second must not share a file
            counter += 1
            self.path = log_dir / f"spore_xl_{label}_{time.strftime('%Y%m%d_%H%M%S')}_{counter}.log"
        self.handle = open(self.path, "a", encoding="utf-8", errors="replace")
        self.handle.write("".join(self.pending))
        self.handle.flush()
        self.pending = []
        os.environ["XL_CONSOLE_LOG"] = str(self.path)
        os.environ["XL_CONSOLE_PARENT_PID"] = str(os.getpid())
        return self.path

    def close(self):
        sys.stdout, sys.stderr = self._original
        if self.handle:
            self.handle.flush()
            self.handle.close()
            self.handle = None
        os.environ.pop("XL_CONSOLE_LOG", None)
        os.environ.pop("XL_CONSOLE_PARENT_PID", None)
