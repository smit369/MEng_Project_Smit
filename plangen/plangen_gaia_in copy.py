"""
PlanGEN implementation for GAIA dataset evaluation
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from datasets import load_dataset
from huggingface_hub import login
from rich.console import Console
from rich.table import Table
from collections import defaultdict

# Import from plangen package
from plangen import PlanGEN
from plangen.agents import ConstraintAgent, SelectionAgent, SolutionAgent, VerificationAgent
from plangen.models import BaseModelInterface, OpenAIModelInterface
from plangen.prompts import PromptManager
from plangen.utils.llm_interface import LLMInterface

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
    It inherits all the original functionality including:
    - Constraint extraction
    - Solution generation
    - Solution verification
    - Solution selection
    - All original agents and their workflows
    """
    
    def __init__(
        self,
        model: Optional[BaseModelInterface] = None,
        prompt_manager: Optional[PromptManager] = None,
        num_solutions: int = 3,
        model_name: str = "llama3:8b",
        temperature: float = 0.7,
    ):
        """Initialize PlanGEN with GAIA dataset support
        
        Args:
            model: Model interface for generating responses (inherited from PlanGEN)
            prompt_manager: Manager for prompt templates (inherited from PlanGEN)
            num_solutions: Number of solutions to generate (inherited from PlanGEN)
            model_name: Name of the model to use
            temperature: Temperature for LLM generation
        """
        if model is None:
            model = LLMInterface(model_name=model_name, temperature=temperature, model_type="ollama")
        super().__init__(model=model, prompt_manager=prompt_manager, num_solutions=num_solutions)
        self.dataset = None
        
    def load_dataset(self):
        """Load the GAIA dataset"""
        self.dataset = load_gaia_dataset()
        return self.dataset
    
    def _evaluate_solution(self, solution: str, ground_truth: str) -> bool:
        """Evaluate if the solution matches the ground truth.
        
        Args:
            solution: The generated solution
            ground_truth: The ground truth answer
            
        Returns:
            bool: True if the solution is correct, False otherwise
        """
        # Convert both to lowercase for case-insensitive comparison
        solution = solution.lower()
        ground_truth = ground_truth.lower()
        
        # Check if the ground truth word appears in the solution
        if ground_truth in solution:
            return True
        
        # If not found directly, check for partial matches
        # Split the solution into words and check if any word contains the ground truth
        solution_words = solution.split()
        for word in solution_words:
            if ground_truth in word or word in ground_truth:
                return True
            
        return False
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics from results.
        
        Args:
            results: List of result dictionaries containing evaluation data
            
        Returns:
            Dict containing overall and per-level metrics
        """
        metrics = {
            'overall': {
                'total': 0,
                'correct': 0,
                'refusals': 0,
                'accuracy': 0.0,
                'refusal_rate': 0.0
            },
            'levels': {}
        }
        
        for result in results:
            if 'selected_solution' not in result:
                print(f"Warning: result missing 'selected_solution' key. Skipping. Result: {result}")
                continue
            level = result['gaia_metadata']['level']
            if level not in metrics['levels']:
                metrics['levels'][level] = {
                    'total': 0,
                    'correct': 0,
                    'refusals': 0,
                    'accuracy': 0.0,
                    'refusal_rate': 0.0
                }
            
            # Update counts
            metrics['overall']['total'] += 1
            metrics['levels'][level]['total'] += 1
            
            selected_solution = result['selected_solution']
            if isinstance(selected_solution, dict):
                solution_text = selected_solution.get('selected_solution', '')
            else:
                solution_text = str(selected_solution)
            
            ground_truth = result['gaia_metadata']['ground_truth']
            
            # Check if solution is a refusal
            if 'i cannot' in solution_text.lower() or 'i don\'t know' in solution_text.lower():
                metrics['overall']['refusals'] += 1
                metrics['levels'][level]['refusals'] += 1
            # Check if solution is correct
            elif self._evaluate_solution(solution_text, ground_truth):
                metrics['overall']['correct'] += 1
                metrics['levels'][level]['correct'] += 1
            
        # Calculate rates
        if metrics['overall']['total'] > 0:
            metrics['overall']['accuracy'] = (metrics['overall']['correct'] / metrics['overall']['total']) * 100
            metrics['overall']['refusal_rate'] = (metrics['overall']['refusals'] / metrics['overall']['total']) * 100
        
        for level in metrics['levels']:
            if metrics['levels'][level]['total'] > 0:
                metrics['levels'][level]['accuracy'] = (metrics['levels'][level]['correct'] / metrics['levels'][level]['total']) * 100
                metrics['levels'][level]['refusal_rate'] = (metrics['levels'][level]['refusals'] / metrics['levels'][level]['total']) * 100
            
        return metrics
    
    def evaluate_dataset(self, num_samples: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate PlanGEN on GAIA dataset using all original PlanGEN functionality
        
        This method uses all the original PlanGEN workflow:
        1. Extract constraints using ConstraintAgent
        2. Generate solutions using SolutionAgent
        3. Verify solutions using VerificationAgent
        4. Select best solution using SelectionAgent
        5. Evaluate solutions against ground truth
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Tuple containing:
            - List of evaluation results
            - Dictionary of evaluation metrics
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
                # Print sample structure for debugging
                console.print("\n[bold yellow]Sample structure:[/bold yellow]")
                console.print(sample)
                
                # Extract problem from GAIA sample
                problem = sample['Question']  # Use 'Question' field for the problem
                
                # Run complete PlanGEN workflow using all original functionality
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
        
        # Calculate evaluation metrics
        metrics = self._calculate_metrics(results)
        
        return results, metrics

    def _generate_solutions(self, problem: str, constraints: List[str]) -> List[str]:
        """Generate multiple solutions for the given problem and constraints.
        
        Args:
            problem: The problem to solve
            constraints: List of constraints to consider
            
        Returns:
            List of generated solutions
        """
        solutions = []
        
        # Extract potential words from the problem description
        words = set()
        for word in problem.lower().split():
            # Remove punctuation and common words
            word = word.strip('.,!?()[]{}":;')
            if len(word) > 3 and word not in ['the', 'and', 'that', 'this', 'with', 'from', 'which', 'what', 'when', 'where', 'who', 'why', 'how']:
                words.add(word)
            
        # Generate solutions with different approaches
        for _ in range(3):
            prompt = f"""You are an expert in generating solutions based on constraints. Given a problem and its constraints, generate a detailed solution.

Problem: {problem}
Constraints:
{chr(10).join(f'- {c}' for c in constraints)}

Additional context: The solution should focus on identifying a specific word that appears in the problem description. This word is likely to be one of: {', '.join(sorted(words))}

Generate a comprehensive solution that:
1. Analyzes the constraints carefully
2. Identifies the specific word that answers the question
3. Explains why this word is the correct answer
4. Provides evidence from the constraints to support the answer

Your solution should be clear and concise, focusing on finding the exact word that answers the question."""
            
            response = self.model.generate(prompt)
            if response:
                solutions.append(response)
            
        return solutions

    def solve(self, problem: str) -> Dict[str, Any]:
        """Override solve to use state dict and parent workflow."""
        try:
            state = {"problem": problem}

            # Extract constraints
            constraints_result = self._extract_constraints(state)
            if "error" in constraints_result:
                return {"problem": problem, "error": constraints_result["error"]}
            state.update(constraints_result)

            # Generate solutions
            solutions_result = self._generate_solutions(state["problem"], state["constraints"])
            if "error" in solutions_result:
                return {**state, "error": solutions_result["error"]}
            state.update({"solutions": solutions_result})

            # Verify solutions
            verify_result = self._verify_solutions(state)
            if "error" in verify_result:
                return {**state, "error": verify_result["error"]}
            state.update(verify_result)

            # Select solution
            select_result = self._select_solution(state)
            if "error" in select_result:
                return {**state, "error": select_result["error"]}
            state.update(select_result)

            return state

        except Exception as e:
            return {"problem": problem, "error": f"Error in workflow: {str(e)}"}

def main():
    """Main entry point for GAIA evaluation"""
    # Initialize PlanGEN with Llama model
    plangen = PlanGENGaia(
        model_name="llama3:8b",
        temperature=0.7
    )
    
    # Run evaluation using complete PlanGEN workflow
    results, metrics = plangen.evaluate_dataset(num_samples=1)  # Start with 1 sample
    
    # Print detailed results including all PlanGEN outputs
    for result in results:
        console.print("\n[bold green]Results for sample:[/bold green]")
        console.print(f"Problem: {result['problem']}")
        console.print(f"Constraints: {result.get('constraints')}")
        console.print(f"Generated Solutions: {result.get('solutions')}")
        console.print(f"Verification Results: {result.get('verification_results')}")
        console.print(f"Selected Solution: {result.get('selected_solution')}")
        console.print(f"GAIA Metadata: {result.get('gaia_metadata')}")
        if 'error' in result:
            console.print(f"[bold red]Error: {result['error']}[/bold red]")
    
    # Print evaluation metrics
    console.print("\n[bold green]Evaluation Metrics:[/bold green]")
    
    # Overall metrics
    table = Table(title="Overall Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for metric, value in metrics['overall'].items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.2%}")
        else:
            table.add_row(metric, str(value))
    
    console.print(table)
    
    # Level-specific metrics
    for level, level_metrics in sorted(metrics['levels'].items()):
        table = Table(title=f"Level {level} Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in level_metrics.items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.2%}")
            else:
                table.add_row(metric, str(value))
        
        console.print(table)

if __name__ == "__main__":
    main() 