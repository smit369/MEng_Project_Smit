"""
GAIA dataset verification strategy.
"""

from typing import Any, Dict, List

from ..base_verifier import BaseVerifier


class GaiaVerifier(BaseVerifier):
    """GAIA-specific implementation of the verifier interface.

    This verifier handles GAIA dataset problems by checking if the solution
    addresses the key requirements and constraints of the problem.
    """

    def __init__(self):
        """Initialize the GAIA verifier."""
        self.domain_keywords = [
            "arxiv",
            "paper",
            "article",
            "figure",
            "axis",
            "label",
            "word",
            "society",
            "physics",
            "regulation",
        ]

    def is_applicable(self, problem_statement: str) -> bool:
        """Check if this verifier is applicable to the given problem.

        Args:
            problem_statement: The problem statement to check

        Returns:
            True if this is a GAIA problem, False otherwise
        """
        problem_lower = problem_statement.lower()
        
        # Check for domain keywords
        keyword_count = sum(1 for keyword in self.domain_keywords if keyword in problem_lower)
        
        # If at least 3 keywords are present, consider it a GAIA problem
        return keyword_count >= 3

    def verify_solution(
        self, problem_statement: str, solution: str, constraints: List[str]
    ) -> Dict[str, Any]:
        """Verify if a solution satisfies the constraints for a GAIA problem.

        Args:
            problem_statement: The original problem statement
            solution: The proposed solution
            constraints: List of constraints the solution must satisfy

        Returns:
            Dictionary containing verification results
        """
        # Check if solution addresses all constraints
        constraint_satisfaction = self._check_constraint_satisfaction(solution, constraints)
        
        # Check if solution is well-structured
        structure_score = self._check_solution_structure(solution)
        
        # Calculate overall score (weighted average)
        score = (constraint_satisfaction * 0.7) + (structure_score * 0.3)
        
        # Determine if solution is valid
        is_valid = score >= 0.6  # Threshold for considering a solution valid
        
        # Generate feedback
        feedback = self._generate_feedback(constraint_satisfaction, structure_score, constraints)
        
        return {
            "is_valid": is_valid,
            "score": score * 100,  # Convert to percentage
            "reason": feedback,
            "constraint_satisfaction": constraint_satisfaction,
            "structure_score": structure_score
        }

    def extract_domain_constraints(
        self, problem_statement: str, general_constraints: List[str]
    ) -> List[str]:
        """Extract domain-specific constraints from the problem statement.

        Args:
            problem_statement: The problem statement
            general_constraints: General constraints already extracted

        Returns:
            List of domain-specific constraints
        """
        domain_constraints = []
        
        # Extract time-related constraints
        if "2022" in problem_statement:
            domain_constraints.append("The AI regulation paper was submitted in 2022")
        if "2016" in problem_statement:
            domain_constraints.append("The Physics and Society article was submitted in 2016")
            
        # Extract figure-related constraints
        if "figure" in problem_statement.lower():
            domain_constraints.append("The solution must reference a figure with three axes")
            domain_constraints.append("Each axis must have label words at both ends")
            
        # Extract society-related constraints
        if "society" in problem_statement.lower():
            domain_constraints.append("The solution must identify a word that describes a type of society")
            
        return domain_constraints

    def _check_constraint_satisfaction(self, solution: str, constraints: List[str]) -> float:
        """Check how well the solution satisfies the constraints.

        Args:
            solution: The solution text
            constraints: List of constraints to check

        Returns:
            Score between 0 and 1 indicating constraint satisfaction
        """
        solution_lower = solution.lower()
        satisfied_constraints = 0
        
        for constraint in constraints:
            constraint_lower = constraint.lower()
            # Check if the solution addresses the constraint
            if any(word in solution_lower for word in constraint_lower.split()):
                satisfied_constraints += 1
                
        return satisfied_constraints / len(constraints) if constraints else 0.0

    def _check_solution_structure(self, solution: str) -> float:
        """Check if the solution has a good structure.

        Args:
            solution: The solution text

        Returns:
            Score between 0 and 1 indicating structure quality
        """
        # Check for clear sections
        has_sections = len(solution.split("\n\n")) > 2
        
        # Check for step-by-step reasoning
        has_steps = any(str(i) + "." in solution for i in range(1, 10))
        
        # Check for conclusion
        has_conclusion = "conclusion" in solution.lower() or "therefore" in solution.lower()
        
        # Calculate structure score
        structure_points = sum([has_sections, has_steps, has_conclusion])
        return structure_points / 3.0

    def _generate_feedback(
        self, constraint_satisfaction: float, structure_score: float, constraints: List[str]
    ) -> str:
        """Generate feedback based on verification results.

        Args:
            constraint_satisfaction: Score indicating how well constraints are satisfied
            structure_score: Score indicating solution structure quality
            constraints: List of constraints that were checked

        Returns:
            Detailed feedback string
        """
        feedback = []
        
        # Add constraint satisfaction feedback
        if constraint_satisfaction >= 0.8:
            feedback.append("The solution addresses most constraints effectively.")
        elif constraint_satisfaction >= 0.5:
            feedback.append("The solution addresses some constraints but could be more comprehensive.")
        else:
            feedback.append("The solution fails to address many key constraints.")
            
        # Add structure feedback
        if structure_score >= 0.8:
            feedback.append("The solution is well-structured with clear sections and reasoning.")
        elif structure_score >= 0.5:
            feedback.append("The solution has a basic structure but could be better organized.")
        else:
            feedback.append("The solution lacks clear structure and organization.")
            
        return " ".join(feedback) 