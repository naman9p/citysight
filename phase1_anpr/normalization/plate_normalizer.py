"""Indian plate normalization + format validation for Phase 1 (Step 9).

Deterministic cleanup and validation only. Crucially, this NEVER substitutes
ambiguous OCR characters (O<->0, I<->1, B<->8, ...): cleanup is limited to
case-folding, dropping separators, and keeping allowed alphanumerics. No fuzzy
correction, edit distance, or reranking.
"""

import re
from dataclasses import dataclass
from typing import Optional


# Defaults used when a config section is not supplied.
DEFAULT_ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DEFAULT_PATTERNS = {
    "standard": r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$",
    "bh_series": r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$",
}
DEFAULT_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}


@dataclass
class NormalizationResult:
    """Outcome of normalizing + validating one plate string."""

    raw_text: str
    normalized_text: str
    is_valid: bool
    format_type: Optional[str]      # "standard", "bh_series", or None
    state_code: Optional[str]       # only when confidently extractable


class PlateNormalizer:
    """Normalize OCR plate text and validate against Indian plate formats."""

    def __init__(self, allowed_chars=DEFAULT_ALLOWED_CHARS, patterns=None,
                 valid_state_codes=None):
        self._allowed = set(allowed_chars)
        patterns = patterns or DEFAULT_PATTERNS
        self._patterns = {name: re.compile(pat) for name, pat in patterns.items()}
        self._state_codes = set(valid_state_codes or DEFAULT_STATE_CODES)

    @classmethod
    def from_config(cls, config):
        norm = config.get("normalization", {})
        return cls(
            allowed_chars=norm.get("allowed_chars", DEFAULT_ALLOWED_CHARS),
            patterns=norm.get("patterns", DEFAULT_PATTERNS),
            valid_state_codes=norm.get("valid_state_codes", DEFAULT_STATE_CODES),
        )

    def normalize_text(self, text: str) -> str:
        """Uppercase, drop separators, and keep only allowed characters.

        Ambiguous characters are preserved as-is — never substituted.
        """
        if not text:
            return ""
        return "".join(ch for ch in text.upper() if ch in self._allowed)

    def normalize(self, text: str) -> NormalizationResult:
        """Full normalize + validate; safe on empty/invalid input."""
        raw = text or ""
        normalized = self.normalize_text(raw)

        if not normalized:
            return NormalizationResult(raw, "", False, None, None)

        format_type = self._match_format(normalized)
        state_code = None
        is_valid = False

        if format_type == "standard":
            candidate_state = normalized[:2]
            if candidate_state in self._state_codes:
                state_code = candidate_state
                is_valid = True
            # A valid standard shape with an unknown RTO code is not confidently
            # a real plate, so it stays invalid with no state_code.
        elif format_type == "bh_series":
            # BH plates are national (no state code).
            is_valid = True

        return NormalizationResult(
            raw_text=raw,
            normalized_text=normalized,
            is_valid=is_valid,
            format_type=format_type if is_valid else None,
            state_code=state_code,
        )

    def _match_format(self, normalized: str) -> Optional[str]:
        for name, pattern in self._patterns.items():
            if pattern.match(normalized):
                return name
        return None
