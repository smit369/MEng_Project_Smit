"""
LLM-based GAIA verifier using prompt decorators and llama3:8b.
"""

import logging
import re
import requests
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..base_verifier import BaseVerifier

logger = logging.getLogger(__name__)

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

@dataclass
class Constraint:
    """Internal representation of a parsed constraint."""
    ctype: ConstraintType
    raw: str
    value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PromptDecorator:
    """Handles prompt decorators for verification"""
    def apply_decorators(self, prompt: str, decorators: set[str]) -> str:
        if not decorators:
            return prompt
            
        enhanced_prompt = prompt
        decorator_instructions = []

        for decorator in decorators:
            if decorator == '+++Reasoning':
                decorator_instructions.append("You must provide detailed reasoning for your assessment.")
            elif decorator == '+++StepByStep':
                decorator_instructions.append("You must evaluate each constraint step-by-step.")
            elif decorator == '+++Critique':
                decorator_instructions.append("You must provide constructive criticism and identify specific issues.")
            elif decorator == '+++CiteSources':
                decorator_instructions.append("You must reference specific parts of the solution to support your assessment.")
                
        if decorator_instructions:
            enhanced_prompt = "IMPORTANT INSTRUCTIONS:\n" + "\n".join(decorator_instructions) + "\n\n" + enhanced_prompt
        return enhanced_prompt

class GaiaVerifierLLM(BaseVerifier):
    """LLM-based GAIA verifier using prompt decorators and llama3:8b."""

    _DOMAIN_KEYWORDS = {
        "arxiv", "dataset", "figure", "word", "code", "syntax", "output",
        "constraints", "question", "paper", "article", "society", "physics",
        "regulation", "label", "programming", "word", "algorithm", "function"
    }

    def __init__(self, model_name: str = "llama3:8b", temperature: float = 0.1):
        """Initialize the LLM-based verifier."""
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_decorator = PromptDecorator()
        
        # Verification decorators to use
        self.verification_decorators = {
            '+++Reasoning',
            '+++StepByStep', 
            '+++Critique',
            '+++CiteSources'
        }

    def is_applicable(self, problem_statement: str) -> bool:
        """Check if this verifier is applicable to the given problem."""
        problem_lower = problem_statement.lower()
        hits = sum(1 for kw in self._DOMAIN_KEYWORDS if kw in problem_lower)
        return hits >= 2

    def extract_domain_constraints(
        self,
        problem_statement: str,
        general_constraints: List[str] | None = None,
        solution: str | None = None
    ) -> List[str]:
        """Extract domain-specific constraints from the problem statement."""
        domain_constraints = []
        
        # Time constraints
        years = re.findall(r"\b(19|20)\d{2}\b", problem_statement)
        for yr in years:
            domain_constraints.append(f"**Time constraint**: The problem references the year {yr}")

        # Figure constraints
        if any(cue in problem_statement.lower() for cue in ["figure", "diagram", "plot", "graph", "axis", "axes", "label"]):
            domain_constraints.extend([
                "**Logical constraint**: The solution must reference any figure or diagram details correctly",
                "**Logical constraint**: Each described axis must be labelled at both ends when applicable"
            ])

        # Society constraints
        if any(cue in problem_statement.lower() for cue in ["society", "community", "group", "organization", "institution"]):
            domain_constraints.append(
                "**Logical constraint**: The solution must correctly identify terminology describing a type of society or community"
            )

        # Code constraints
        if any(cue in problem_statement.lower() for cue in ["code", "programming", "syntax", "function", "output", "result"]):
            domain_constraints.extend([
                "**Logical constraint**: The solution must follow valid programming syntax",
                "**Logical constraint**: The solution must produce the expected output or behaviour"
            ])

        return domain_constraints

    def verify_solution(
        self,
        problem_statement: str,
        solution: str,
        constraints: List[str],
    ) -> Dict[str, Any]:
        """Verify if a solution satisfies the constraints using LLM."""
        logger.debug(f"LLM-based verification for problem: {problem_statement[:100]}...")
        
        try:
            # Create verification prompt
            verification_prompt = self._create_verification_prompt(problem_statement, solution, constraints)
            
            # Apply prompt decorators
            enhanced_prompt = self.prompt_decorator.apply_decorators(verification_prompt, self.verification_decorators)
            
            # Get LLM response
            llm_response = self._call_llm(enhanced_prompt)
            
            # Parse the response
            result = self._parse_llm_response(llm_response, constraints)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM verification: {str(e)}")
            return self._fallback_verification(solution, constraints)

    def _create_verification_prompt(self, problem: str, solution: str, constraints: List[str]) -> str:
        """Create a comprehensive verification prompt."""
        
        constraints_text = "\n".join([f"- {c}" for c in constraints])
        
        prompt = f"""You are an expert evaluator for GAIA benchmark problems. Your task is to verify if a given solution satisfies all the constraints for a specific problem.

PROBLEM:
{problem}

SOLUTION TO VERIFY:
{solution}

CONSTRAINTS TO CHECK:
{constraints_text}

Please evaluate the solution against each constraint and provide:

1. **Overall Assessment**: Does the solution satisfy the constraints? (Yes/No)
2. **Constraint-by-Constraint Analysis**: For each constraint, indicate if it's satisfied (✓) or not satisfied (✗) and explain why
3. **Score**: Provide a score from 0-100 based on constraint satisfaction
4. **Detailed Reasoning**: Explain your assessment with specific references to the solution
5. **Suggestions**: What could be improved if the solution doesn't fully satisfy the constraints?

Format your response as:
ASSESSMENT: [Yes/No]
SCORE: [0-100]
REASONING: [Detailed explanation]
CONSTRAINT_ANALYSIS:
- Constraint 1: [✓/✗] [Explanation]
- Constraint 2: [✓/✗] [Explanation]
...
SUGGESTIONS: [Improvement suggestions]"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM using Ollama API."""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise

    def _parse_llm_response(self, response: str, constraints: List[str]) -> Dict[str, Any]:
        """Parse the LLM response to extract verification results."""
        try:
            # Extract assessment
            assessment_match = re.search(r'ASSESSMENT:\s*(Yes|No)', response, re.IGNORECASE)
            is_valid = assessment_match.group(1).lower() == 'yes' if assessment_match else False
            
            # Extract score
            score_match = re.search(r'SCORE:\s*(\d+)', response)
            score = int(score_match.group(1)) if score_match else 50
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.*?)(?=CONSTRAINT_ANALYSIS:|SUGGESTIONS:|$)', response, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No detailed reasoning provided"
            
            # Count satisfied constraints
            satisfied_count = len(re.findall(r'✓', response))
            total_constraints = len(constraints)
            constraint_satisfaction = satisfied_count / total_constraints if total_constraints > 0 else 0.0
            
            # Calculate structure score based on response quality
            structure_score = self._calculate_structure_score(response)
            
            return {
                'is_valid': is_valid,
                'score': float(score),
                'reason': reasoning,
                'constraint_satisfaction': constraint_satisfaction,
                'structure_score': structure_score,
                'parsed_constraints': [{'raw': c, 'satisfied': '✓' in response} for c in constraints],
                'failed_matches': [{'constraint': c, 'type': 'logical', 'value': None} 
                                 for c in constraints if '✓' not in response]
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return self._fallback_verification("", constraints)

    def _calculate_structure_score(self, response: str) -> float:
        """Calculate structure score based on response quality."""
        score = 0.0
        
        # Check for structured sections
        if 'ASSESSMENT:' in response:
            score += 0.25
        if 'SCORE:' in response:
            score += 0.25
        if 'REASONING:' in response:
            score += 0.25
        if 'CONSTRAINT_ANALYSIS:' in response:
            score += 0.25
            
        return score

    def _fallback_verification(self, solution: str, constraints: List[str]) -> Dict[str, Any]:
        """Fallback verification when LLM fails."""
        return {
            'is_valid': False,
            'score': 0.0,
            'reason': 'Verification failed due to LLM error',
            'constraint_satisfaction': 0.0,
            'structure_score': 0.0,
            'parsed_constraints': [{'raw': c, 'satisfied': False} for c in constraints],
            'failed_matches': [{'constraint': c, 'type': 'logical', 'value': None} for c in constraints]
        } 