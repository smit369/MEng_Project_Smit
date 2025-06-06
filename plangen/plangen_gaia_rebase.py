"""
PlanGEN implementation for GAIA dataset evaluation
"""

import os
from typing import Dict, Any, List, Optional
from datasets import load_dataset
from huggingface_hub import login
from rich.console import Console
import json

# Import from plangen package
from plangen import PlanGEN
from plangen.agents import ConstraintAgent, SelectionAgent, SolutionAgent, VerificationAgent
from plangen.models import BaseModelInterface, OpenAIModelInterface
from plangen.prompts import PromptManager
from plangen.utils.llm_interface import LLMInterface
from plangen.algorithms.rebase import REBASE

console = Console()

def load_gaia_dataset():
    """Load the GAIA dataset with authentication"""
    # Set Hugging Face token
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


    login(token=os.getenv('HF_TOKEN'))
    
    console.print("[bold blue]Loading GAIA dataset from Hugging Face...[/bold blue]")
    dataset = load_dataset("gaia-benchmark/GAIA", '2023_all', token=os.getenv('HF_TOKEN'), trust_remote_code=True)
    
    # Print dataset structure for debugging
    console.print("\n[bold yellow]Dataset structure:[/bold yellow]")
    console.print(dataset['validation'].features)
    
    return dataset['validation']  # Using validation set for evaluation

class PlanGENGaia(PlanGEN):
    """PlanGEN implementation for GAIA dataset evaluation
    
    This class extends the original PlanGEN implementation to work with the GAIA dataset.
    It uses the REBASE algorithm for solution generation and refinement.
    """
    
    def __init__(
        self,
        model: Optional[BaseModelInterface] = None,
        prompt_manager: Optional[PromptManager] = None,
        num_solutions: int = 3,
        model_name: str = "llama3:8b",
        temperature: float = 0.7,
        max_iterations: int = 5,
        improvement_threshold: float = 0.1,
    ):
        """Initialize PlanGEN with GAIA dataset support
        
        Args:
            model: Model interface for generating responses
            prompt_manager: Manager for prompt templates
            num_solutions: Number of solutions to generate
            model_name: Name of the model to use
            temperature: Temperature for LLM generation
            max_iterations: Maximum number of REBASE refinement iterations
            improvement_threshold: Minimum score improvement to continue refining
        """
        if model is None:
            model = LLMInterface(model_name=model_name, temperature=temperature, model_type="ollama")
        super().__init__(model=model, prompt_manager=prompt_manager, num_solutions=num_solutions)
        
        # Initialize REBASE algorithm
        self.rebase = REBASE(
            max_iterations=max_iterations,
            improvement_threshold=improvement_threshold,
            domain="gaia",
            llm_interface=model,
            constraint_agent=self.constraint_agent,
            verification_agent=self.verification_agent
        )
        
        self.dataset = None
        
    def load_dataset(self):
        """Load the GAIA dataset"""
        self.dataset = load_gaia_dataset()
        return self.dataset
    
    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve a problem using REBASE algorithm
        
        Args:
            problem: The problem statement to solve
            
        Returns:
            Dictionary containing:
            - constraints: List of extracted constraints
            - solutions: List of generated solutions
            - verification_results: Results from solution verification
            - selected_solution: Best solution after verification
            - rebase_metadata: Metadata from REBASE algorithm
        """
        # Extract constraints
        constraints = self.constraint_agent.run(problem)
        
        # Generate solutions using REBASE
        best_plan, score, rebase_metadata = self.rebase.run(problem)
        
        # Verify the solution
        verification_result = self.verification_agent.run(problem, constraints, best_plan)
        
        # Select the best solution (in this case, it's the REBASE solution)
        selected_solution = {
            'selected_solution': best_plan,
            'score': score,
            'verification_result': verification_result
        }
        
        return {
            'problem': problem,
            'constraints': constraints,
            'solutions': [best_plan],  # REBASE gives us one refined solution
            'verification_results': [verification_result],
            'selected_solution': selected_solution,
            'rebase_metadata': rebase_metadata
        }
    
    def evaluate_dataset(self, num_samples: int = None) -> List[Dict[str, Any]]:
        """Evaluate PlanGEN on GAIA dataset using REBASE algorithm
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            List of evaluation results containing:
            - REBASE results (constraints, solutions, verification, selection)
            - GAIA-specific metadata
        """
        if self.dataset is None:
            self.load_dataset()
            
        # Get samples to evaluate
        samples = self.dataset
        if num_samples is not None:
            samples = samples.select(range(min(num_samples, len(samples))))
            
        results = []
        for idx, sample in enumerate(samples):
            console.print(f"\n[bold blue]Processing sample {idx + 1}/{len(samples)}[/bold blue]")
            
            try:
                # Extract problem from GAIA sample
                problem = sample['Question']
                
                # Run REBASE algorithm
                result = self.solve(problem)
                
                # Add GAIA metadata
                result['gaia_metadata'] = {
                    'level': sample.get('Level', ''),
                    'ground_truth': sample.get('Final answer', ''),
                    'task_id': sample.get('task_id', ''),
                    'file_name': sample.get('file_name', ''),
                    'file_path': sample.get('file_path', ''),
                    'annotator_metadata': sample.get('Annotator Metadata', {})
                }
                
                results.append(result)
                
            except Exception as e:
                console.print(f"[bold red]Error processing sample {idx + 1}: {str(e)}[/bold red]")
                results.append({
                    'error': str(e),
                    'problem': problem if 'problem' in locals() else None
                })
            
        return results

def main():
    """Main entry point for GAIA evaluation"""
    # Initialize PlanGEN with Llama model
    plangen = PlanGENGaia(
        model_name="llama3:8b",
        temperature=0.7,
        max_iterations=5,
        improvement_threshold=0.1
    )
    
    # Run evaluation using REBASE algorithm
    results = plangen.evaluate_dataset()
    
    # Save results to a JSON file
    output_file = "plangen_gaia_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    console.print(f"\n[bold green]Results saved to {output_file}[/bold green]")
    
    # Print detailed results including all PlanGEN outputs
    for result in results:
        console.print("\n[bold green]Results for sample:[/bold green]")
        console.print(f"Problem: {result['problem']}")
        console.print(f"Constraints: {result.get('constraints')}")
        console.print(f"Generated Solutions: {result.get('solutions')}")
        console.print(f"Verification Results: {result.get('verification_results')}")
        console.print(f"Selected Solution: {result.get('selected_solution')}")
        console.print(f"REBASE Metadata: {result.get('rebase_metadata')}")
        console.print(f"GAIA Metadata: {result.get('gaia_metadata')}")
        if 'error' in result:
            console.print(f"[bold red]Error: {result['error']}[/bold red]")

if __name__ == "__main__":
    main() 