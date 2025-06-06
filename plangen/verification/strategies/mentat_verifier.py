from ..base_verifier import BaseVerifier
from typing import List, Dict, Any

class MentatVerifier(BaseVerifier):
    """Verifier for Mentat-style problems."""
    
    def is_applicable(self, problem_statement: str) -> bool:
        """Check if this verifier is applicable to the given problem."""
        # Check for Mentat-specific keywords
        mentat_keywords = ["mentat", "dune", "human computer", "human calculator"]
        return any(keyword in problem_statement.lower() for keyword in mentat_keywords)
    
    def verify_solution(self, problem_statement: str, solution: str, constraints: List[str]) -> Dict[str, Any]:
        """Verify if the solution satisfies the constraints."""
        # Minimal logic: valid if all constraints are mentioned in the solution
        is_valid = all(constraint.lower() in solution.lower() for constraint in constraints)
        score = 100 if is_valid else 0
        reason = "All constraints satisfied." if is_valid else "Some constraints not satisfied."
        return {"is_valid": is_valid, "score": score, "reason": reason}
    
    def extract_domain_constraints(self, problem_statement: str, general_constraints: List[str]) -> List[str]:
        """Extract domain-specific constraints from the problem."""
        # Minimal logic: return general constraints unchanged
        return general_constraints
    
    def get_feedback(self, solution: str, constraints: list) -> str:
        """Generate feedback based on verification results."""
        feedback = []
        
        # Check each constraint
        for constraint in constraints:
            if constraint.lower() not in solution.lower():
                feedback.append(f"Solution does not address constraint: {constraint}")
        
        if not feedback:
            feedback.append("Solution satisfies all constraints.")
            
        return "\n".join(feedback) 