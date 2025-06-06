"""
PlanGEN implementation for GAIA dataset
"""

import csv
import json
import os
from typing import Dict, List, Optional

from plangen.agents.base_agent import BaseAgent
from plangen.utils.llm_interface import LLMInterface


class GAIAAgent(BaseAgent):
    """Agent for handling GAIA dataset tasks."""

    def __init__(
        self,
        llm_interface: Optional[LLMInterface] = None,
        model_name: str = "llama3:8b",
        temperature: float = 0.7,
        system_message: Optional[str] = None,
    ):
        """Initialize the GAIA agent.

        Args:
            llm_interface: Optional LLM interface to use
            model_name: Name of the model to use if llm_interface is not provided
            temperature: Temperature for LLM generation
            system_message: System message to set context for the agent
        """
        super().__init__(
            llm_interface=llm_interface,
            model_name=model_name,
            temperature=temperature,
            system_message=system_message or "You are a helpful AI assistant working with the GAIA dataset.",
        )

    def run(self, task: Dict) -> Dict:
        """Run the agent on a GAIA task.

        Args:
            task: GAIA task dictionary containing 'Question' and 'Ground Truth'

        Returns:
            Dictionary containing the agent's response
        """
        # Extract task information
        question = task.get("Question", "")
        ground_truth = task.get("Ground Truth", "")

        # Generate prompt based on task
        prompt = self._generate_task_prompt(question)

        # Get response from LLM
        response = self._call_llm(prompt)

        return {
            "question": question,
            "ground_truth": ground_truth,
            "model_prediction": response,
        }

    def _generate_task_prompt(self, question: str) -> str:
        """Generate a prompt for the given question.

        Args:
            question: The question to answer

        Returns:
            Formatted prompt
        """
        template = """
Question: {question}

Please provide a detailed and accurate answer to this question. Consider the following:
1. What is the question asking for?
2. What information or resources are needed to answer it?
3. What steps should be taken to find the answer?
4. What potential challenges might arise in finding the answer?

Your response:
"""
        return self._generate_prompt(
            template,
            question=question,
        )


def load_gaia_dataset(file_path: str) -> List[Dict]:
    """Load GAIA dataset from a CSV file.

    Args:
        file_path: Path to the GAIA dataset CSV file

    Returns:
        List of task dictionaries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GAIA dataset file not found: {file_path}")

    tasks = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)

    return tasks


def main():
    """Main function to run the GAIA agent."""
    # Initialize the agent
    agent = GAIAAgent()

    # Load GAIA dataset
    dataset_path = "gaia_eval_table.csv"  # Update with actual path
    tasks = load_gaia_dataset(dataset_path)

    # Process each task
    results = []
    for task in tasks:
        result = agent.run(task)
        results.append(result)

    # Save results
    output_path = "gaia_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main() 