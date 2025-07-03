"""
Robust verification agent for PlanGEN with enhanced error handling.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from ..verification import BaseVerifier, VerifierFactory

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RobustVerificationAgent(BaseAgent):
    """Robust verification agent with enhanced error handling for solutions against constraints."""

    def __init__(
        self,
        llm_interface=None,
        model_name: str = "llama3:8b",
        temperature: float = 0.7,
        verifier: Optional[BaseVerifier] = None
    ):
        """Initialize the robust verification agent.

        Args:
            llm_interface: LLM interface for generating responses
            model_name: Name of the model to use if llm_interface is not provided
            temperature: Temperature for LLM generation
            verifier: Optional specific verifier to use. If None, will auto-detect.
        """
        super().__init__(llm_interface, model_name, temperature)
        self.verifier = verifier
        self.verifier_factory = VerifierFactory()
        logger.debug(f"Initialized RobustVerificationAgent with model: {model_name}, temperature: {temperature}")

    def run(
        self,
        problem_statement: str,
        constraints: List[str],
        plan: str,
    ) -> Tuple[str, float]:
        """Verify a plan against constraints with robust error handling.

        Args:
            problem_statement: Original problem statement
            constraints: List of constraints
            plan: Plan to verify

        Returns:
            Tuple of (feedback, score)
        """
        logger.debug(f"Starting robust verification for problem: {problem_statement[:100]}...")
        logger.debug(f"Plan to verify: {plan[:100]}...")
        logger.debug(f"Constraints: {constraints}")

        # Ensure constraints is always a list
        if constraints is None:
            constraints = []
        elif not isinstance(constraints, list):
            logger.warning(f"Constraints is not a list: {type(constraints)}, converting to empty list")
            constraints = []

        # Get appropriate verifier
        try:
            if self.verifier:
                logger.debug("Using provided verifier")
                verifier = self.verifier
            else:
                logger.debug("Auto-detecting verifier")
                verifier = self.verifier_factory.get_verifier(problem_statement)
                logger.debug(f"Selected verifier: {verifier.__class__.__name__}")
        except Exception as e:
            logger.error(f"Error getting verifier: {str(e)}")
            # Return default failure result
            return "Verification: FAIL\nReason: Failed to get appropriate verifier\nScore: 0", 0.0

        # Extract domain-specific constraints with robust error handling
        logger.debug("Extracting domain-specific constraints")
        try:
            domain_constraints = verifier.extract_domain_constraints(
                problem_statement, constraints
            )
            # Ensure domain_constraints is always a list
            if domain_constraints is None:
                domain_constraints = []
            elif not isinstance(domain_constraints, list):
                logger.warning(f"extract_domain_constraints returned {type(domain_constraints)}, converting to list")
                domain_constraints = []
        except Exception as e:
            logger.error(f"Error extracting domain constraints: {str(e)}")
            domain_constraints = []
            
        # Safely combine constraints
        try:
            all_constraints = constraints + domain_constraints
            logger.debug(f"Combined constraints: {all_constraints}")
        except Exception as e:
            logger.error(f"Error combining constraints: {str(e)}")
            all_constraints = constraints if constraints else []

        # Verify the solution with robust error handling
        logger.debug("Verifying solution")
        try:
            results = verifier.verify_solution(problem_statement, plan, all_constraints)
            logger.debug(f"Verification results: {results}")
        except Exception as e:
            logger.error(f"Error in verify_solution: {str(e)}")
            results = {
                'is_valid': False,
                'score': 0.0,
                'reason': f"Verification failed: {str(e)}"
            }

        # Format feedback with safe access to results
        try:
            is_valid = results.get('is_valid', False)
            reason = results.get('reason', 'No reason provided')
            score = results.get('score', 0.0)
            
            feedback = f"Verification: {'PASS' if is_valid else 'FAIL'}\n"
            feedback += f"Reason: {reason}\n"
            feedback += f"Score: {score}"
            logger.debug(f"Formatted feedback: {feedback}")

            # Ensure score is a valid float
            try:
                score_float = float(score)
            except (ValueError, TypeError):
                logger.warning(f"Invalid score value: {score}, using 0.0")
                score_float = 0.0

            return feedback, score_float
            
        except Exception as e:
            logger.error(f"Error formatting verification feedback: {str(e)}")
            return "Verification: FAIL\nReason: Error formatting results\nScore: 0", 0.0 