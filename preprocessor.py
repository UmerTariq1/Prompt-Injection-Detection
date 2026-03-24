"""
preprocessor.py
---------------
Cleans the raw input before any detection runs.

Attackers love obfuscation: zero-width characters, homoglyphs,
Base64-encoded payloads, ROT13, leetspeak, etc.
We strip all of that here so every downstream stage sees plain text.

Order matters:
  1. Strip invisible / control characters
  2. NFKC Unicode normalisation (collapses fullwidth → ASCII, ligatures, etc.)
  3. Homoglyph → ASCII  (Cyrillic "а" → Latin "a", etc.)
  4. Decode encoded payloads (Base64, hex, URL, HTML entities, ROT13)
  5. Normalise leetspeak
  6. Collapse whitespace
"""

import base64
import html
import logging
import re
import unicodedata
import urllib.parse
from codecs import decode as codecs_decode

logger = logging.getLogger(__name__)


# ── Homoglyph map ─────────────────────────────────────────────────────────────
# Covers the most common cross-script confusables used in prompt injection.
# Source: Unicode Consortium confusables.txt (security-relevant subset).
HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic → Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    # Greek → Latin
    "α": "a", "β": "b", "ε": "e", "ο": "o", "ρ": "p", "υ": "u",
    "ν": "v", "Α": "A", "Β": "B", "Ε": "E", "Ο": "O", "Ρ": "P",
    # Fullwidth Latin (already handled by NFKC, kept as safety net)
    "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e",
    # Mathematical bold / italic variants
    "𝐚": "a", "𝐛": "b", "𝐜": "c", "𝐢": "i", "𝐧": "n",
    "𝘢": "a", "𝘣": "b", "𝘤": "c",
}

# ── Leetspeak substitutions ────────────────────────────────────────────────────
LEET_MAP: dict[str, str] = {
    "@": "a", "4": "a",
    "3": "e",
    "!": "i", "1": "i",
    "0": "o",
    "$": "s", "5": "s",
    "7": "t",
    "+": "t",
}

# ── Zero-width / invisible Unicode categories to strip ────────────────────────
# Cf = Format characters (includes zero-width joiners, soft hyphens, etc.)
_INVISIBLE_PATTERN = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]"
)

# ── Base64 detection heuristic ────────────────────────────────────────────────
_B64_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{4}){3,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")


def _strip_invisible(text: str) -> str:
    """Remove zero-width, bidirectional override, and other invisible characters."""
    return _INVISIBLE_PATTERN.sub("", text)


def _nfkc_normalize(text: str) -> str:
    """NFKC decomposition: fullwidth → ASCII, ligatures → base chars, etc."""
    return unicodedata.normalize("NFKC", text)


def _apply_homoglyphs(text: str) -> str:
    """Replace known cross-script confusables with their ASCII equivalents."""
    return "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)


def _try_decode_base64(text: str) -> str:
    """Attempt to decode Base64 segments found in the text."""
    def replace_b64(match: re.Match) -> str:
        candidate = match.group(0)
        # Pad to a multiple of 4
        padded = candidate + "=" * (-len(candidate) % 4)
        try:
            decoded = base64.b64decode(padded).decode("utf-8", errors="ignore")
            # Only replace if the decoded result looks like readable text
            if decoded.isprintable() and len(decoded) > 3:
                logger.debug("Decoded Base64 segment: %r → %r", candidate, decoded)
                return decoded
        except Exception:
            pass
        return candidate

    return _B64_PATTERN.sub(replace_b64, text)


def _try_decode_hex(text: str) -> str:
    """Decode hex-encoded strings like \\x69\\x67\\x6e\\x6f\\x72\\x65."""
    hex_pattern = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")
    def replace_hex(match: re.Match) -> str:
        try:
            raw = bytes.fromhex(match.group(0).replace("\\x", ""))
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return match.group(0)
    return hex_pattern.sub(replace_hex, text)


def _try_decode_rot13(text: str) -> str:
    """
    ROT13 is sometimes used to smuggle injection keywords.
    We decode and append — keeping original so we don't lose context.
    Only triggered if the decoded text contains injection-related words.
    """
    decoded = codecs_decode(text, "rot_13")
    injection_keywords = {"ignore", "system", "override", "jailbreak", "instructions"}
    if any(kw in decoded.lower() for kw in injection_keywords):
        logger.debug("ROT13 decoded suspicious content — appending decoded version.")
        return text + " " + decoded
    return text


def _try_decode_url(text: str) -> str:
    """Decode URL-percent-encoded strings."""
    try:
        decoded = urllib.parse.unquote(text)
        if decoded != text:
            logger.debug("URL-decoded text.")
        return decoded
    except Exception:
        return text


def _try_decode_html_entities(text: str) -> str:
    """Unescape HTML entities like &lt; &#105;gnore etc."""
    return html.unescape(text)


def _normalize_leet(text: str) -> str:
    """
    Convert leetspeak digits/symbols to letters.
    Applied conservatively — only on tokens that look like they contain leet.
    Whole-word numbers (e.g. "3 items") are left alone.
    """
    # Only apply inside words that mix letters and leet chars
    leet_word = re.compile(r"\b[a-zA-Z0-9@$!+]{3,}\b")

    def replace_leet(match: re.Match) -> str:
        word = match.group(0)
        # Skip if it's a plain number
        if word.isdigit():
            return word
        # Skip if it contains no leet substitution targets
        if not any(ch in LEET_MAP for ch in word):
            return word
        return "".join(LEET_MAP.get(ch, ch) for ch in word)

    return leet_word.sub(replace_leet, text)


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> tuple[str, list[str]]:
    """
    Run the full normalization pipeline on raw input text.

    Returns
    -------
    cleaned : str
        The normalized text ready for detection.
    transformations : list[str]
        Human-readable list of which transformations were applied
        (useful for debugging / audit logs).
    """
    transformations: list[str] = []
    original = text

    text = _strip_invisible(text)
    if text != original:
        transformations.append("stripped invisible/zero-width chars")

    text_after_nfkc = _nfkc_normalize(text)
    if text_after_nfkc != text:
        transformations.append("NFKC normalised")
        text = text_after_nfkc

    text_after_hg = _apply_homoglyphs(text)
    if text_after_hg != text:
        transformations.append("homoglyphs replaced")
        text = text_after_hg

    text_after_html = _try_decode_html_entities(text)
    if text_after_html != text:
        transformations.append("HTML entities decoded")
        text = text_after_html

    text_after_url = _try_decode_url(text)
    if text_after_url != text:
        transformations.append("URL-percent-encoding decoded")
        text = text_after_url

    text_after_hex = _try_decode_hex(text)
    if text_after_hex != text:
        transformations.append("hex escape sequences decoded")
        text = text_after_hex

    text_after_b64 = _try_decode_base64(text)
    if text_after_b64 != text:
        transformations.append("Base64 segments decoded")
        text = text_after_b64

    text_after_rot = _try_decode_rot13(text)
    if text_after_rot != text:
        transformations.append("ROT13 decoded (suspicious keywords found)")
        text = text_after_rot

    text_after_leet = _normalize_leet(text)
    if text_after_leet != text:
        transformations.append("leetspeak normalised")
        text = text_after_leet

    text = _collapse_whitespace(text)

    return text, transformations
