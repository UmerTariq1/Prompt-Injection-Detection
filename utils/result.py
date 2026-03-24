"""
result.py
---------
Shared data structure returned by every detection stage.
Keeps the pipeline clean — each stage speaks the same language.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    SAFE      = "SAFE"
    INJECTION = "INJECTION"
    UNCERTAIN = "UNCERTAIN"   # only possible from Stage 2; triggers Stage 3


@dataclass
class DetectionResult:
    verdict:     Verdict
    score:       float           # 0.0 = definitely safe, 1.0 = definitely injection
    stage:       int             # which stage produced this verdict (0-3)
    reason:      str             # human-readable explanation
    latency_ms:  float = 0.0
    matched_patterns: list[str] = field(default_factory=list)  # Stage 1 only
    raw_scores:  Optional[dict]  = None                         # Stage 2/3 logits

    def is_injection(self) -> bool:
        return self.verdict == Verdict.INJECTION

    def __str__(self) -> str:
        patterns = ""
        if self.matched_patterns:
            patterns = f"\n   Patterns : {', '.join(self.matched_patterns[:5])}"
        return (
            f"[Stage {self.stage}] {self.verdict.value}  "
            f"(score={self.score:.3f}, {self.latency_ms:.1f}ms)\n"
            f"   Reason   : {self.reason}"
            f"{patterns}"
        )
