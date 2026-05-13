"""Compat imports : `ParsingService` + alias `TextParsingService`."""

from __future__ import annotations

from app.parsing.base import ParsingService
from app.parsing.heuristic_text import HeuristicTextParser

TextParsingService = HeuristicTextParser

__all__ = ["HeuristicTextParser", "ParsingService", "TextParsingService"]
