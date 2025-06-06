"""
Solution agent for PlanGEN
"""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent


class SolutionAgent(BaseAgent):
    """Agent for generating solutions based on constraints."""

    def __init__(
        self,
        llm_interface=None,
        model_name: str = "llama3:8b",
        temperature: float = 0.7,
    ):
        """Initialize the solution agent.

        Args:
            llm_interface: LLM interface for generating responses
            model_name: Name of the model to use if llm_interface is not provided
            temperature: Temperature for LLM generation
        """
        super().__init__(llm_interface, model_name, temperature)
        
        self.solution_prompt_template = (
            "You are an expert in generating solutions based on constraints. "
            "Given a problem and its constraints, generate a detailed solution.\n\n"
            "Problem: {problem}\n"
            "Constraints:\n{constraints}\n\n"
            "Generate a comprehensive solution that addresses all constraints."
        )

    def run(self, problem: str, constraints: str, num_solutions: int = 3) -> List[str]:
        """Generate multiple solutions for a problem.

        Args:
            problem: Problem statement
            constraints: Extracted constraints
            num_solutions: Number of solutions to generate

        Returns:
            List of generated solutions
        """
        prompt = self._generate_prompt(
            self.solution_prompt_template,
            problem=problem,
            constraints=constraints
        )

        # Generate multiple solutions
        solutions = []
        for i in range(num_solutions):
            solution = self._call_llm(prompt=prompt)
            solutions.append(solution)

        return solutions
