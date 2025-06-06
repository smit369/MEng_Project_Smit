"""
Verification agent for PlanGEN.
"""

from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from ..verification import BaseVerifier, VerifierFactory


class VerificationAgent(BaseAgent):
    """Agent for verifying solutions against constraints."""

    def __init__(
        self,
        llm_interface=None,
        model_name: str = "llama3:8b",
        temperature: float = 0.7,
        verifier: Optional[BaseVerifier] = None
    ):
        """Initialize the verification agent.

        Args:
            llm_interface: LLM interface for generating responses
            model_name: Name of the model to use if llm_interface is not provided
            temperature: Temperature for LLM generation
            verifier: Optional specific verifier to use. If None, will auto-detect.
        """
        super().__init__(llm_interface, model_name, temperature)
        self.verifier = verifier
        self.verifier_factory = VerifierFactory()

    def run(
        self,
        problem_statement: str,
        constraints: List[str],
        plan: str,
    ) -> Tuple[str, float]:
        """Verify a plan against constraints.

        Args:
            problem_statement: Original problem statement
            constraints: List of constraints
            plan: Plan to verify

        Returns:
            Tuple of (feedback, score)
        """
        # Get appropriate verifier
        verifier = self.verifier or self.verifier_factory.get_verifier(
            problem_statement
        )

        # Extract domain-specific constraints
        domain_constraints = verifier.extract_domain_constraints(
            problem_statement, constraints
        )
        all_constraints = constraints + domain_constraints

        # Verify the solution
        results = verifier.verify_solution(problem_statement, plan, all_constraints)

        # Format feedback
        feedback = f"Verification: {'PASS' if results['is_valid'] else 'FAIL'}\n"
        feedback += f"Reason: {results['reason']}\n"
        feedback += f"Score: {results['score']}"

        return feedback, float(results["score"])
