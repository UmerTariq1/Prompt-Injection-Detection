"""
stage1_heuristics.py
--------------------
Stage 1 — Regex Heuristics

Fast pattern matching using scored regex rules.
Organized into categories, each with a weight.
If the cumulative score crosses a threshold → immediate INJECTION verdict.
If the score is very low → immediate SAFE verdict.
Otherwise → pass to Stage 2.

Pattern categories (highest to lowest weight):
  CRITICAL  (1.0) — unmistakable injection signatures
  HIGH      (0.7) — strong indicators
  MEDIUM    (0.4) — suspicious but ambiguous
  LOW       (0.15)— weak signals (context matters)
"""

import logging
import re
import time
from dataclasses import dataclass

from utils.config import CFG
from utils.result import DetectionResult, Verdict

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    name:   str
    regex:  re.Pattern
    weight: float
    description: str


def _p(name: str, pattern: str, weight: float, description: str = "") -> Pattern:
    """Helper to compile a pattern with IGNORECASE + DOTALL."""
    return Pattern(
        name=name,
        regex=re.compile(pattern, re.IGNORECASE | re.DOTALL),
        weight=weight,
        description=description,
    )


# ── Pattern Library ────────────────────────────────────────────────────────────

CRITICAL_PATTERNS: list[Pattern] = [
    _p("ignore_previous",
       r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?|constraints?)",
       1.0, "Classic override: 'ignore previous instructions'"),

    _p("disregard_instructions",
       r"disregard\s+(all\s+)?(previous|prior|above|your|the)?\s*(instructions?|prompts?|rules?|guidelines?|constraints?)",
       1.0, "Disregard instructions variant"),

    _p("new_instructions_override",
       r"(your\s+new|from\s+now\s+on|henceforth|hereafter).{0,40}(instructions?|rules?|role|persona|task)",
       0.9, "'Your new instructions are...' override"),

    _p("system_prompt_override",
       r"\[?(SYSTEM|INST|INSTRUCTION|ADMIN|ROOT|OVERRIDE|UPDATE)\]?\s*:?\s*(ignore|forget|disregard|override|you\s+are\s+now)",
       1.0, "Fake system marker with override command"),

    _p("jailbreak_dan",
       r"\b(DAN|do\s+anything\s+now|jailbreak(ed)?|developer\s+mode|god\s+mode)\b",
       1.0, "DAN / jailbreak keyword"),

    _p("you_are_now",
       r"you\s+are\s+now\s+(a|an|the)\s+\w+",
       0.9, "Persona hijacking: 'you are now a...'"),

    _p("act_as",
       r"\bact\s+as\s+(a|an|the|if\s+you\s+(are|were))\b",
       0.85, "Role-play override: 'act as a/an'"),

    _p("pretend_you_are",
       r"\b(pretend|imagine|roleplay|role.play)\s+(you\s+(are|were)|to\s+be)\b",
       0.85, "Persona injection via pretend/imagine"),

    _p("forget_everything",
       r"forget\s+(everything|all|your\s+(previous|prior|original))\s*(you\s+know|instructions?|training|rules?)?",
       0.95, "'Forget everything' override"),

    _p("confidentiality_leak",
       r"(reveal|show|print|output|repeat|tell\s+me|what\s+is)\s+(your|the)\s+(system\s+)?prompt",
       1.0, "System prompt extraction attempt"),

    _p("token_smuggling_end",
       r"(</?(system|instruction|prompt|context|human|assistant)>)",
       0.9, "Fake XML/HTML token injection"),
]

HIGH_PATTERNS: list[Pattern] = [
    _p("do_not_follow",
       r"do\s+not\s+(follow|obey|adhere\s+to|comply\s+with)\s+(the\s+)?(instructions?|rules?|guidelines?|constraints?)",
       0.7, "Explicit instruction negation"),

    _p("override_safety",
       r"(override|bypass|disable|circumvent|neutralize)\s+(safety|security|content|filter|restriction|guard|policy|alignment)",
       0.7, "Safety override / bypass attempt"),

    _p("without_restrictions",
       r"without\s+(any\s+)?(restrictions?|limitations?|filters?|rules?|constraints?|guidelines?|censorship)",
       0.7, "'Without restrictions' freedom request"),

    _p("respond_as",
       r"respond\s+(as|like)\s+(if\s+you\s+(are|were)|a|an)",
       0.65, "Respond-as persona injection"),

    _p("simulate_mode",
       r"\b(simulation\s+mode|training\s+mode|test\s+mode|debug\s+mode|maintenance\s+mode|unrestricted\s+mode)\b",
       0.7, "Fake mode activation"),

    _p("above_is_untrusted",
       r"(the\s+)?(above|previous|prior)\s+(text|content|message|prompt)\s+is\s+(untrusted|malicious|fake|injection)",
       0.65, "Indirect injection preamble"),

    _p("eval_execute",
       r"\b(eval|exec|execute|run)\s*\(",
       0.7, "Code execution injection"),

    _p("base64_suspicious",
       r"base64\s*(decode|encoded|string)",
       0.65, "Explicit Base64 decode request"),

    _p("repeat_verbatim",
       r"(repeat|print|say|output)\s+(verbatim|word\s+for\s+word|exactly|literally)",
       0.6, "Verbatim output request (often used for prompt leaking)"),
]

MEDIUM_PATTERNS: list[Pattern] = [
    _p("new_persona",
       r"(take\s+on|adopt|assume)\s+(the\s+)?(role|persona|identity|character)\s+of",
       0.4, "Persona adoption request"),

    _p("instruction_follows",
       r"(follow\s+these|here\s+are\s+(your\s+)?(new\s+)?|updated)\s+instructions?",
       0.4, "New instruction injection preamble"),

    _p("anything_is_allowed",
       r"(anything\s+is\s+(allowed|permitted|acceptable|fine)|no\s+(rules?|restrictions?|limits?))",
       0.4, "Anything-goes framing"),

    _p("sudo_admin",
       r"\b(sudo|admin|root|superuser|administrator)\s+(mode|access|command|override|prompt)\b",
       0.4, "Privilege escalation framing"),

    _p("actually_an_ai",
       r"you\s+are\s+(actually|really|just|only)\s+(a|an)\s+(AI|LLM|language\s+model|chatbot|assistant)",
       0.35, "Model identity confusion attempt"),

    _p("confidential_data_request",
       r"(show|reveal|expose|leak|extract)\s+(confidential|private|sensitive|internal|hidden|secret)",
       0.45, "Confidential data extraction"),

    _p("harmful_content_request",
       r"(how\s+to|instructions?\s+for|steps?\s+to)\s+(make|create|build|synthesize|hack|attack|bomb|weapon)",
       0.45, "Harmful content request"),

    _p("translate_then_execute",
       r"translate\s+(this|the\s+following).{0,60}(and\s+then|then)\s+(execute|run|follow)",
       0.4, "Translate-then-execute chained attack"),
]

LOW_PATTERNS: list[Pattern] = [
    _p("hypothetically",
       r"\b(hypothetically|theoretically|for\s+(educational|research|academic)\s+purposes?|in\s+a\s+fictional\s+(scenario|world))\b",
       0.15, "Hypothetical framing (weak signal)"),

    _p("in_this_story",
       r"\b(in\s+this\s+(story|narrative|scenario|game|fiction|novel)|as\s+a\s+character)\b",
       0.15, "Fictional framing (weak signal)"),

    _p("two_language_switch",
       r"[\u4e00-\u9fff\u3040-\u30ff\u0600-\u06ff\u0900-\u097f]{5,}",
       0.15, "Non-Latin script block (possible multilingual evasion)"),
]

ALL_PATTERNS: list[Pattern] = (
    CRITICAL_PATTERNS + HIGH_PATTERNS + MEDIUM_PATTERNS + LOW_PATTERNS
)


def _compute_score(text: str) -> tuple[float, list[str]]:
    """
    Run all patterns against the text.
    Returns a score in [0, 1] and a list of matched pattern names.
    Score is capped at 1.0 using additive accumulation with diminishing returns.
    """
    raw_score    = 0.0
    matched      = []

    for pattern in ALL_PATTERNS:
        if pattern.regex.search(text):
            raw_score += pattern.weight
            matched.append(pattern.name)

    # Sigmoid-like cap: high scores stay near 1.0, don't exceed it
    capped = min(raw_score, 1.0)
    return capped, matched


def check(text: str) -> DetectionResult:
    """
    Run heuristic checks on preprocessed text.

    Returns a DetectionResult with:
      - INJECTION if score >= heuristic_inject_threshold
      - SAFE      if heuristic_safe_threshold > 0 and score <= heuristic_safe_threshold
      - UNCERTAIN otherwise (Stage 2 will decide)

    heuristic_safe_threshold <= 0 disables the SAFE fast-path so “clean” prompts
    still go to the ML stages.
    """
    t0 = time.perf_counter()

    score, matched = _compute_score(text)

    latency = (time.perf_counter() - t0) * 1000

    if score >= CFG.heuristic_inject_threshold:
        return DetectionResult(
            verdict=Verdict.INJECTION,
            score=score,
            stage=1,
            reason=f"Heuristic patterns matched (score={score:.3f})",
            latency_ms=latency,
            matched_patterns=matched,
        )

    if CFG.heuristic_safe_threshold > 0 and score <= CFG.heuristic_safe_threshold:
        return DetectionResult(
            verdict=Verdict.SAFE,
            score=score,
            stage=1,
            reason="No significant injection patterns found",
            latency_ms=latency,
            matched_patterns=matched,
        )

    # Uncertain — let the ML classifier decide
    return DetectionResult(
        verdict=Verdict.UNCERTAIN,
        score=score,
        stage=1,
        reason=f"Ambiguous heuristic score ({score:.3f}) — escalating to ML classifier",
        latency_ms=latency,
        matched_patterns=matched,
    )
