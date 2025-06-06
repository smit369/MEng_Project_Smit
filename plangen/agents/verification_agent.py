"""
Verification agent for PlanGEN.
"""

import logging
import ipdb
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from ..verification import BaseVerifier, VerifierFactory

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        logger.debug(f"Initialized VerificationAgent with model: {model_name}, temperature: {temperature}")

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
        logger.debug(f"Starting verification for problem: {problem_statement[:100]}...")
        logger.debug(f"Plan to verify: {plan[:100]}...")
        logger.debug(f"Constraints: {constraints}")

        # Get appropriate verifier
        if self.verifier:
            logger.debug("Using provided verifier")
            verifier = self.verifier
        else:
            logger.debug("Auto-detecting verifier")
            verifier = self.verifier_factory.get_verifier(problem_statement)
            logger.debug(f"Selected verifier: {verifier.__class__.__name__}")

        # Extract domain-specific constraints
        logger.debug("Extracting domain-specific constraints")
        domain_constraints = verifier.extract_domain_constraints(
            problem_statement, constraints
        )
        all_constraints = constraints + domain_constraints
        logger.debug(f"Combined constraints: {all_constraints}")

        # Verify the solution
        logger.debug("Verifying solution")
        results = verifier.verify_solution(problem_statement, plan, all_constraints)
        logger.debug(f"Verification results: {results}")

        # Format feedback
        feedback = f"Verification: {'PASS' if results['is_valid'] else 'FAIL'}\n"
        feedback += f"Reason: {results['reason']}\n"
        feedback += f"Score: {results['score']}"
        logger.debug(f"Formatted feedback: {feedback}")

        return feedback, float(results["score"])
