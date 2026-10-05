import logging
import os
import re
import sys
import warnings

NOISE_PATTERNS = [
    re.compile(r"self\.slidingWindow:"),
    re.compile(r"scores:"),
    re.compile(r"^\s*\(\d+,\)\s*$"),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"GPU is unavailable|Using CPU"),
]


class _FilteredStdout:
    def __init__(self, stream):
        self.stream = stream
        self.dropped = False

    def write(self, text):
        if text == "\n" and self.dropped:
            self.dropped = False
            return 0
        self.dropped = bool(text.strip()) and any(p.search(text.strip()) for p in NOISE_PATTERNS)
        if self.dropped:
            return 0
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()


def apply_silence():
    warnings.filterwarnings("ignore")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_WARNINGS"] = "0"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    os.environ["TQDM_DISABLE"] = "1"
    for name in ["torch", "codecarbon", "TSB_AD", "river", "deep_river", "urllib3"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    sys.stdout = _FilteredStdout(sys.stdout)
