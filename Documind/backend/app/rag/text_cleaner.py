"""Text cleaner and normalizer for parsed document text.

EDUCATIONAL ARCHITECTURE NOTES (VIVA):
--------------------------------------
Raw text extracted from PDFs and Word documents is fraught with layout artifacts:
1. Broken Hyphenation: Line wraps such as "trans-\nportation" split semantic words
   across lines, causing sub-word tokenizers (like Gemini's SentencePiece/BPE) to produce
   disjoint tokens ("trans", "-", "portation"), corrupting embedding representations.
2. Irregular Whitespaces: Mixed tabs, multi-spaces, non-breaking spaces (\u00a0),
   and excessive empty lines artificially bloat context windows without adding value.
3. Control Characters: Residual form feeds, vertical tabs, and null bytes can trigger
   malformed JSON strings or parser crashes in downstream GenAI APIs.

Preserving clean double-newlines (`\\n\\n`) retains paragraph semantic boundaries,
which is essential for boundary-aware chunking.
"""

import re


def clean_text(text: str) -> str:
    """Normalizes raw extracted text while preserving intentional paragraph structure.

    Args:
        text: Raw document text string.

    Returns:
        Cleaned, normalized text suitable for chunking and embedding.
    """
    if not text:
        return ""

    # Replace Windows carriage returns and form feeds
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")

    # Rejoin hyphenated line-end splits (e.g. "concur-\nrent" -> "concurrent")
    cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)

    # Replace non-breaking spaces and exotic whitespace with standard space
    cleaned = cleaned.replace("\u00a0", " ").replace("\u200b", "")

    # Strip control characters (ASCII 0-31 except tab \t and newline \n)
    cleaned = "".join(ch for ch in cleaned if ch in ("\t", "\n") or ord(ch) >= 32)

    # Normalize multiple inline spaces/tabs to a single space, but preserve newlines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Collapse sequences of 3 or more newlines down to 2 (paragraph break)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Clean leading and trailing whitespace on individual lines
    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)

    # Final trim
    return cleaned.strip()
