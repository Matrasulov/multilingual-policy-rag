from __future__ import annotations

import logging
import re
from typing import Iterable


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# ---------
# Text filters (TOC / template junk)
# ---------

def is_toc_like(text: str) -> bool:
    tl = text.lower().strip()
    if "table of contents" in tl or tl in {"contents", "index"}:
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 8:
        ends_with_num = sum(1 for ln in lines[:20] if ln and ln[-1].isdigit())
        if ends_with_num / min(len(lines), 20) >= 0.5:
            return True
    return False


DEFAULT_STOP_PATTERNS = (
    r"_{3,}",                 # placeholders like ________
    r"\(adjust to align",      # template guidance
)


def keep_text(text: str, stop_patterns: Iterable[str] = DEFAULT_STOP_PATTERNS) -> bool:
    if is_toc_like(text):
        return False
    if any(re.search(p, text, re.IGNORECASE) for p in stop_patterns):
        return False
    return True
