"""
Revised GAIA verifier that can score arbitrary question‑answering outputs
against the heterogeneous natural‑language constraints that appear in the
GAIA benchmark.

Main improvements
-----------------
* Registry pattern for constraint checkers
* Vector similarity fallback for semantic matching
* Schema-validated output contract
* Feature flags for unsafe operations
* Comprehensive logging
* External configuration for mappings
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

from pydantic import BaseModel, Field

from ..base_verifier import BaseVerifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Type variables for generic decorators
T = TypeVar('T')
CheckerFunc = Callable[[str, str], bool]

# ──────────────────────────────────────────────────────────────────────────────
# Configuration and schemas
# ──────────────────────────────────────────────────────────────────────────────
class ConstraintType(str, Enum):
    """Canonical constraint categories."""
    TIME = "time"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"
    LOGICAL = "logical"
    PHYSICAL = "physical"
    SPECIFIC = "specific"
    NUMERICAL = "numerical"
    BOOLEAN = "boolean"
    SQL = "sql"
    CODE = "code"


class VerificationResult(BaseModel):
    """Schema-validated output contract for all verifiers."""
    is_valid: bool = Field(..., description="Whether the solution passes all constraints")
    score: float = Field(..., ge=0, le=100, description="Overall score (0-100)")
    reason: str = Field(..., description="Human-readable feedback")
    constraint_satisfaction: float = Field(..., ge=0, le=1, description="Constraint satisfaction score (0-1)")
    structure_score: float = Field(..., ge=0, le=1, description="Structure quality score (0-1)")
    parsed_constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Parsed constraint details")
    failed_matches: List[Dict[str, Any]] = Field(default_factory=list, description="Details of failed constraint matches")


@dataclass
class Constraint:
    """Internal representation of a parsed constraint."""
    ctype: ConstraintType
    raw: str
    value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VerifierConfig(BaseModel):
    """Configuration for the GAIA verifier."""
    enable_unsafe_ops: bool = Field(False, description="Enable code execution, regex compilation, etc.")
    use_vector_similarity: bool = Field(True, description="Use embeddings for semantic matching")
    similarity_threshold: float = Field(0.8, ge=0, le=1, description="Minimum similarity score")
    pass_threshold: float = Field(0.6, ge=0, le=1, description="Minimum passing score")
    constraint_weight: float = Field(0.7, ge=0, le=1, description="Weight for constraint satisfaction")
    structure_weight: float = Field(0.3, ge=0, le=1, description="Weight for structure quality")
    keyword_threshold: int = Field(2, ge=0, description="Minimum number of domain keywords to match")


# ──────────────────────────────────────────────────────────────────────────────
# Registry for constraint checkers
# ──────────────────────────────────────────────────────────────────────────────
class ConstraintCheckerRegistry:
    """Registry for constraint type checkers."""
    _checkers: Dict[ConstraintType, List[CheckerFunc]] = {}

    @classmethod
    def register(cls, ctype: ConstraintType) -> Callable[[CheckerFunc], CheckerFunc]:
        """Decorator to register a checker function for a constraint type."""
        def decorator(func: CheckerFunc) -> CheckerFunc:
            if ctype not in cls._checkers:
                cls._checkers[ctype] = []
            cls._checkers[ctype].append(func)
            return func
        return decorator

    @classmethod
    def get_checkers(cls, ctype: ConstraintType) -> List[CheckerFunc]:
        """Get all registered checkers for a constraint type."""
        return cls._checkers.get(ctype, [])


# ──────────────────────────────────────────────────────────────────────────────
# Main verifier
# ──────────────────────────────────────────────────────────────────────────────
class GaiaVerifier(BaseVerifier):
    """GAIA‑specific implementation able to evaluate free‑text constraints."""

    # Load configuration from JSON/YAML
    _config_path = Path(__file__).parent / "gaia_config.json"
    _config = VerifierConfig()

    # Regular expressions
    YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
    PARENTHETICAL_RE = re.compile(r"['\"]([^'\"]+)['\"]")
    OUTPUT_RE = re.compile(r'output\s+"([^"]+)"')

    # Cue word sets
    FIGURE_CUES = {"figure", "diagram", "plot", "graph", "axis", "axes", "label"}
    SOCIETY_CUES = {"society", "community", "group", "organization", "institution"}
    CODE_CUES = {"code", "programming", "syntax", "function", "output", "result"}

    # Domain keywords for problem recognition
    _DOMAIN_KEYWORDS = {
        "arxiv", "dataset", "figure", "word", "code", "syntax", "output",
        "constraints", "question", "paper", "article", "society", "physics",
        "regulation", "label", "programming", "word", "algorithm", "function",
        "variable", "loop", "condition", "statement", "expression", "operator",
        "data", "structure", "type", "class", "method", "parameter", "argument",
        "return", "value", "result", "error", "exception", "debug", "test",
        "verify", "validate", "check", "assert", "compare", "match", "pattern"
    }

    def __init__(self):
        """Initialize the verifier with configuration."""
        self._load_config()
        self._setup_vector_similarity()

    def _load_config(self) -> None:
        """Load configuration from file."""
        if self._config_path.exists():
            with open(self._config_path) as f:
                self._config = VerifierConfig(**json.load(f))

    def _setup_vector_similarity(self) -> None:
        """Initialize vector similarity if enabled."""
        if self._config.use_vector_similarity:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.warning("Vector similarity disabled: sentence-transformers not installed")
                self._config.use_vector_similarity = False

    def is_applicable(self, problem_statement: str) -> bool:
        """Check if this verifier is applicable to the given problem."""
        problem_lower = problem_statement.lower()
        hits = sum(1 for kw in self._DOMAIN_KEYWORDS if kw in problem_lower)
        return hits >= self._config.keyword_threshold

    def extract_domain_constraints(
        self,
        problem_statement: str,
        general_constraints: List[str] | None = None,
        solution: str | None = None
    ) -> List[str]:
        """Extract domain-specific constraints from the problem statement."""
        domain_constraints: List[str] = []

        # Time constraints
        years = set(self.YEAR_RE.findall(problem_statement))
        for yr in years:
            domain_constraints.append(
                f"**Time constraint**: The problem references the year {yr}"
            )

        # Figure constraints
        if any(cue in problem_statement.lower() for cue in self.FIGURE_CUES):
            domain_constraints.extend([
                "**Logical constraint**: The solution must reference any figure or diagram details correctly",
                "**Logical constraint**: Each described axis must be labelled at both ends when applicable"
            ])

        # Society constraints
        if any(cue in problem_statement.lower() for cue in self.SOCIETY_CUES):
            domain_constraints.append(
                "**Logical constraint**: The solution must correctly identify terminology describing a type of society or community"
            )

        # Code constraints
        if any(cue in problem_statement.lower() for cue in self.CODE_CUES):
            domain_constraints.extend([
                "**Logical constraint**: The solution must follow valid programming syntax",
                "**Logical constraint**: The solution must produce the expected output or behaviour"
            ])

        # Solution-specific constraints
        if solution:
            self._extract_solution_constraints(problem_statement, solution, domain_constraints)

        return domain_constraints

    def _extract_solution_constraints(
        self,
        problem_statement: str,
        solution: str,
        constraints: List[str]
    ) -> None:
        """Extract additional constraints based on the solution."""
        # Output requirements
        if "output" in problem_statement.lower():
            output_match = self.OUTPUT_RE.search(problem_statement.lower())
            if output_match:
                expected_output = output_match.group(1)
                constraints.append(
                    f"**Logical constraint**: The solution must produce the exact output: {expected_output}"
                )

        # Parenthetical requirements
        for match in self.PARENTHETICAL_RE.finditer(problem_statement):
            quoted = match.group(1)
            constraints.append(
                f"**Specific constraint**: The solution must include '{quoted}'"
            )

    def verify_solution(
        self,
        problem_statement: str,
        solution: str,
        constraints: List[str],
    ) -> Dict[str, Any]:
        """Verify if a solution satisfies the constraints."""
        logger.debug("Verifying solution – constraints provided: %d", len(constraints))

        parsed = self._parse_constraints(constraints)
        constraint_score = self._check_constraint_satisfaction(solution, parsed)
        structure_score = self._score_structure(solution)

        final_score = (
            constraint_score * self._config.constraint_weight +
            structure_score * self._config.structure_weight
        )

        is_valid = final_score >= self._config.pass_threshold
        feedback = self._build_feedback(constraint_score, structure_score)

        result = VerificationResult(
            is_valid=is_valid,
            score=round(final_score * 100, 2),
            reason=feedback,
            constraint_satisfaction=round(constraint_score, 2),
            structure_score=round(structure_score, 2),
            parsed_constraints=[c.__dict__ for c in parsed],
            failed_matches=self._get_failed_matches(solution, parsed)
        )

        return result.dict()

    def _parse_constraints(self, lines: List[str]) -> List[Constraint]:
        """Parse raw constraints into structured format."""
        parsed: List[Constraint] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract constraint type and value
            ctype = self._extract_constraint_type(line)
            value = self._extract_constraint_value(line)
            
            parsed.append(Constraint(
                ctype=ctype,
                raw=line,
                value=value
            ))

        logger.debug("Parsed %d constraint(s)", len(parsed))
        return parsed

    def _extract_constraint_type(self, line: str) -> ConstraintType:
        """Extract the constraint type from a line."""
        # Try to match known constraint types
        for ctype in ConstraintType:
            if ctype.value in line.lower():
                return ctype
        return ConstraintType.LOGICAL

    def _extract_constraint_value(self, line: str) -> Optional[str]:
        """Extract the constraint value from a line."""
        # Look for colon-separated value
        if ":" in line:
            return line.split(":", 1)[1].strip()
        return None

    def _check_constraint_satisfaction(
        self,
        solution: str,
        constraints: List[Constraint],
    ) -> float:
        """Check if constraints are satisfied using registered checkers."""
        satisfied = 0
        for cons in constraints:
            if self._check_constraint(solution, cons):
                satisfied += 1
        return satisfied / len(constraints) if constraints else 0.0

    def _check_constraint(self, solution: str, constraint: Constraint) -> bool:
        """Check a single constraint using registered checkers."""
        # Try registered checkers first
        for checker in ConstraintCheckerRegistry.get_checkers(constraint.ctype):
            if checker(solution, constraint.value or ""):
                return True

        # Fall back to vector similarity if enabled
        if self._config.use_vector_similarity:
            return self._check_vector_similarity(solution, constraint)

        # Default to token overlap
        return self._check_token_overlap(solution, constraint)

    def _check_vector_similarity(self, solution: str, constraint: Constraint) -> bool:
        """Check constraint satisfaction using vector similarity."""
        if not hasattr(self, '_model'):
            return False

        solution_embedding = self._model.encode(solution)
        constraint_embedding = self._model.encode(constraint.raw)
        similarity = self._model.similarity(solution_embedding, constraint_embedding)
        
        return similarity >= self._config.similarity_threshold

    def _check_token_overlap(self, solution: str, constraint: Constraint) -> bool:
        """Check constraint satisfaction using token overlap."""
        solution_tokens = set(re.findall(r"\w+", solution.lower()))
        constraint_tokens = set(re.findall(r"\w+", constraint.raw.lower()))
        
        if not constraint_tokens:
            return True
            
        overlap = len(solution_tokens & constraint_tokens)
        return overlap / len(constraint_tokens) >= 0.5

    def _get_failed_matches(
        self,
        solution: str,
        constraints: List[Constraint]
    ) -> List[Dict[str, Any]]:
        """Get details of failed constraint matches."""
        failed = []
        for cons in constraints:
            if not self._check_constraint(solution, cons):
                failed.append({
                    "constraint": cons.raw,
                    "type": cons.ctype.value,
                    "value": cons.value
                })
        return failed

    def _score_structure(self, solution: str) -> float:
        """Score the structure of the solution."""
        lines = solution.splitlines()
        sections = sum(1 for line in lines if line.strip() and line.strip().endswith(":"))
        has_steps = bool(re.search(r"^\s*\d+\.\s", solution, re.M))
        has_conclusion = bool(re.search(r"\btherefore\b|\bconclusion\b", solution, re.I))
        has_code = "```" in solution or re.search(r"`[^`]+`", solution)

        points = sum([sections > 1, has_steps, has_conclusion, has_code])
        return points / 4.0

    def _build_feedback(self, constraint_score: float, structure_score: float) -> str:
        """Build feedback message based on scores."""
        msg = []
        if constraint_score >= 0.8:
            msg.append("Most constraints satisfied.")
        elif constraint_score >= 0.5:
            msg.append("Some constraints satisfied; verify remaining requirements.")
        else:
            msg.append("Many constraints unsatisfied.")

        if structure_score >= 0.8:
            msg.append("Answer is well structured.")
        elif structure_score >= 0.5:
            msg.append("Consider adding clearer structure.")
        else:
            msg.append("Answer lacks structure.")

        return " ".join(msg)


# ──────────────────────────────────────────────────────────────────────────────
# Registered constraint checkers
# ──────────────────────────────────────────────────────────────────────────────
@ConstraintCheckerRegistry.register(ConstraintType.SPECIFIC)
def check_specific_constraint(solution: str, value: str) -> bool:
    """Check if a specific value is present in the solution."""
    if not value:
        return True
    value = value.strip().lower().strip("\"'")
    return value in solution.lower()


@ConstraintCheckerRegistry.register(ConstraintType.TIME)
def check_time_constraint(solution: str, value: str) -> bool:
    """Check if time-related constraints are satisfied."""
    if not value:
        return True
    year_match = re.search(r"\b(19|20)\d{2}\b", value)
    if year_match:
        return year_match.group(0) in solution
    return True


@ConstraintCheckerRegistry.register(ConstraintType.CODE)
def check_code_constraint(solution: str, value: str) -> bool:
    """Check if code-related constraints are satisfied."""
    if not value:
        return True
    return "```" in solution or re.search(r"`[^`]+`", solution)
