"""
Selection agent for PlanGEN
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .base_agent import BaseAgent


class Solution(BaseModel):
    """Model for a solution and its verification."""

    text: str = Field(..., description="The solution text")
    verification: Any = Field(..., description="Verification results for the solution")

    @model_validator(mode='after')
    def validate_verification(self) -> 'Solution':
        """Validate and normalize verification field."""
        if not isinstance(self.verification, (str, dict)):
            self.verification = str(self.verification)
        return self

    def get_verification_text(self) -> str:
        """Convert verification results to a string representation."""
        if isinstance(self.verification, str):
            return self.verification
        elif isinstance(self.verification, dict):
            # Convert dict to a readable string format
            return "\n".join(f"{k}: {v}" for k, v in self.verification.items())
        return str(self.verification)


class SelectionAgent(BaseAgent):
    """Agent for selecting the best solution based on verification results."""

    def __init__(
        self,
        llm_interface=None,
        model_name: str = "llama3:8b",
        temperature: float = 0.7,
    ):
        """Initialize the selection agent.

        Args:
            llm_interface: LLM interface for generating responses
            model_name: Name of the model to use if llm_interface is not provided
            temperature: Temperature for LLM generation
        """
        super().__init__(llm_interface, model_name, temperature)
        
        self.selection_prompt_template = (
            "You are an expert in evaluating and selecting the best solution. "
            "Given multiple solutions and their verification results, select the best one.\n\n"
            "Solutions:\n{solutions}\n\n"
            "Select the best solution and explain your reasoning."
        )

    def run(
        self, solutions: List[str], verification_results: List[Any]
    ) -> Dict[str, Any]:
        """Select the best solution based on verification results.

        Args:
            solutions: List of solutions
            verification_results: List of verification results (can be strings or dicts)

        Returns:
            Dictionary with the best solution and selection reasoning
        """
        # Prepare solution objects for the prompt
        solution_objects = [
            Solution(text=solution, verification=verification)
            for solution, verification in zip(solutions, verification_results)
        ]

        prompt = self._generate_prompt(
            self.selection_prompt_template,
            solutions="\n\n".join(
                f"Solution {i+1}:\n{s.text}\nVerification: {s.get_verification_text()}"
                for i, s in enumerate(solution_objects)
            )
        )

        selection_reasoning = self._call_llm(prompt=prompt)

        # Extract the selected solution index (assuming it's mentioned in the reasoning)
        # This is a simple heuristic; in practice, you might want a more robust approach
        selected_index = 0
        for i, solution in enumerate(solutions):
            if (
                f"Solution {i+1}" in selection_reasoning
                and "best" in selection_reasoning.lower()
            ):
                selected_index = i
                break

        return {
            "selected_solution": solutions[selected_index],
            "selection_reasoning": selection_reasoning,
            "selected_index": selected_index,
        }

    def select_best_solution(
        self, solutions: List[str], verification_results: List[Any]
    ) -> Dict[str, Any]:
        """Alias for run method to maintain compatibility with the rest of the codebase.

        Args:
            solutions: List of solutions
            verification_results: List of verification results (can be strings or dicts)

        Returns:
            Dictionary with the best solution and selection reasoning
        """
        return self.run(solutions, verification_results)
