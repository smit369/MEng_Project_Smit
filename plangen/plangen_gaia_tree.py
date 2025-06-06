"""
PlanGEN implementation for GAIA dataset evaluation using Tree of Thought
"""

import os
import time
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
from plangen.algorithms.tree_of_thought import TreeOfThought

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
    It uses the Tree of Thought algorithm for solution generation and refinement.
    """
    
    def __init__(
        self,
        model: Optional[BaseModelInterface] = None,
        prompt_manager: Optional[PromptManager] = None,
        num_solutions: int = 3,
        model_name: str = "llama3:8b",
        temperature: float = 0.7,
        branching_factor: int = 3,
        max_depth: int = 5,
        beam_width: int = 2,
    ):
        """Initialize PlanGEN with GAIA dataset support
        
        Args:
            model: Model interface for generating responses
            prompt_manager: Manager for prompt templates
            num_solutions: Number of solutions to generate
            model_name: Name of the model to use
            temperature: Temperature for LLM generation
            branching_factor: Number of branches to explore at each node
            max_depth: Maximum depth of the tree
            beam_width: Number of paths to keep at each level
        """
        if model is None:
            model = LLMInterface(model_name=model_name, temperature=temperature, model_type="ollama")
        super().__init__(model=model, prompt_manager=prompt_manager, num_solutions=num_solutions)
        
        # Initialize Tree of Thought algorithm
        self.tree = TreeOfThought(
            branching_factor=branching_factor,
            max_depth=max_depth,
            beam_width=beam_width,
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
        """Solve a problem using Tree of Thought
        
        Args:
            problem: The problem statement to solve
            
        Returns:
            Dictionary containing:
            - constraints: List of extracted constraints
            - solutions: List of generated solutions
            - verification_results: Results from solution verification
            - selected_solution: Best solution after verification
            - tree_metadata: Metadata from Tree of Thought
        """
        # Extract constraints
        constraints = self.constraint_agent.run(problem)
        
        # Generate solutions using Tree of Thought
        best_plan, score, tree_metadata = self.tree.run(problem)
        
        # Verify the solution
        verification_result = self.verification_agent.run(problem, constraints, best_plan)
        
        # Select the best solution (in this case, it's the Tree of Thought solution)
        selected_solution = {
            'selected_solution': best_plan,
            'score': score,
            'verification_result': verification_result
        }
        
        return {
            'problem': problem,
            'constraints': constraints,
            'solutions': [best_plan],  # Tree of Thought gives us one refined solution
            'verification_results': [verification_result],
            'selected_solution': selected_solution,
            'tree_metadata': tree_metadata
        }
    
    def evaluate_dataset(self, num_samples: int = None) -> List[Dict[str, Any]]:
        """Evaluate PlanGEN on GAIA dataset using Tree of Thought
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            List of evaluation results containing:
            - Tree of Thought results (constraints, solutions, verification, selection)
            - GAIA-specific metadata
        """
        if self.dataset is None:
            self.load_dataset()
            
        # Get samples to evaluate
        samples = self.dataset
        if num_samples is not None:
            samples = samples.select(range(min(num_samples, len(samples))))
            
        results = []
        total_start_time = time.time()
        
        for idx, sample in enumerate(samples):
            sample_start_time = time.time()
            console.print(f"\n[bold blue]Processing sample {idx + 1}/{len(samples)}[/bold blue]")
            
            try:
                # Extract problem from GAIA sample
                problem = sample['Question']
                
                # Run Tree of Thought
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
                
                # Calculate sample processing time
                sample_time = (time.time() - sample_start_time) / 60  # Convert to minutes
                result['processing_time_minutes'] = sample_time
                console.print(f"[bold green]Sample processing time: {sample_time:.2f} minutes[/bold green]")
                
                results.append(result)
                
            except Exception as e:
                console.print(f"[bold red]Error processing sample {idx + 1}: {str(e)}[/bold red]")
                results.append({
                    'error': str(e),
                    'problem': problem if 'problem' in locals() else None,
                    'processing_time_minutes': (time.time() - sample_start_time) / 60
                })
        
        # Calculate total runtime
        total_time = (time.time() - total_start_time) / 60  # Convert to minutes
        console.print(f"\n[bold green]Total runtime: {total_time:.2f} minutes[/bold green]")
        
        # Add runtime information to results
        results.append({
            'runtime_metadata': {
                'total_runtime_minutes': total_time,
                'average_time_per_sample_minutes': total_time / len(samples),
                'total_samples_processed': len(samples)
            }
        })
            
        return results

def main():
    """Main entry point for GAIA evaluation"""
    # Initialize PlanGEN with Llama model
    plangen = PlanGENGaia(
        model_name="llama3:8b",
        temperature=0.7,
        branching_factor=3,
        max_depth=5,
        beam_width=2
    )
    
    # Run evaluation using Tree of Thought
    results = plangen.evaluate_dataset()
    
    # Save results to a JSON file
    output_file = "plangen_gaia_tree_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    console.print(f"\n[bold green]Results saved to {output_file}[/bold green]")
    
    # Print detailed results including all PlanGEN outputs
    for result in results:
        if 'runtime_metadata' in result:
            # Print runtime summary
            console.print("\n[bold yellow]Runtime Summary:[/bold yellow]")
            console.print(f"Total Runtime: {result['runtime_metadata']['total_runtime_minutes']:.2f} minutes")
            console.print(f"Average Time per Sample: {result['runtime_metadata']['average_time_per_sample_minutes']:.2f} minutes")
            console.print(f"Total Samples Processed: {result['runtime_metadata']['total_samples_processed']}")
            continue
            
        console.print("\n[bold green]Results for sample:[/bold green]")
        console.print(f"Problem: {result['problem']}")
        console.print(f"Constraints: {result.get('constraints')}")
        console.print(f"Generated Solutions: {result.get('solutions')}")
        console.print(f"Verification Results: {result.get('verification_results')}")
        console.print(f"Selected Solution: {result.get('selected_solution')}")
        console.print(f"Tree Metadata: {result.get('tree_metadata')}")
        console.print(f"GAIA Metadata: {result.get('gaia_metadata')}")
        console.print(f"Processing Time: {result.get('processing_time_minutes', 'N/A'):.2f} minutes")
        if 'error' in result:
            console.print(f"[bold red]Error: {result['error']}[/bold red]")

if __name__ == "__main__":
    main() 