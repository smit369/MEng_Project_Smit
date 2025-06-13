"""
PlanGEN implementation for GAIA dataset evaluation
"""

import os
from typing import Dict, Any, List, Optional
from datasets import load_dataset
from huggingface_hub import login
from rich.console import Console
import json
import re

# Import from plangen package
from plangen import PlanGEN
from plangen.agents import ConstraintAgent, SelectionAgent, SolutionAgent, VerificationAgent
from plangen.models import BaseModelInterface, OpenAIModelInterface
from plangen.prompts import PromptManager
from plangen.utils.llm_interface import LLMInterface
from plangen.algorithms.rebase import REBASE

console = Console()

class PromptDecorator:
    """Handles prompt decorators for solution generation"""
    
    def __init__(self):
        self.active_decorators = set()
        self.chat_scope = False
        
    def parse_decorators(self, prompt: str) -> tuple[str, set[str]]:
        """Parse decorators from a prompt and return clean prompt and decorators"""
        decorators = set()
        lines = prompt.split('\n')
        clean_lines = []
        
        for line in lines:
            if line.startswith('+++'):
                decorator = line.strip()
                if decorator == '+++ChatScope':
                    self.chat_scope = True
                elif decorator == '+++MessageScope':
                    self.chat_scope = False
                elif decorator == '+++Clear':
                    self.active_decorators.clear()
                elif decorator.startswith('+++Clear('):
                    # Parse specific decorators to clear
                    decs_to_clear = re.findall(r'\+\+\+[A-Za-z]+', decorator)
                    for dec in decs_to_clear:
                        self.active_decorators.discard(dec)
                else:
                    decorators.add(decorator)
                    if self.chat_scope:
                        self.active_decorators.add(decorator)
            else:
                clean_lines.append(line)
                
        return '\n'.join(clean_lines), decorators
    
    def apply_decorators(self, prompt: str, decorators: set[str]) -> str:
        """Apply decorators to the prompt"""
        if not decorators:
            return prompt
            
        enhanced_prompt = prompt
        
        # Add decorator instructions at the beginning
        decorator_instructions = []
        
        for decorator in decorators:
            if decorator == '+++Reasoning':
                decorator_instructions.append("You must begin your response with a detailed explanation of your reasoning and logic.")
            elif decorator == '+++StepByStep':
                decorator_instructions.append("You must structure your response into a sequence of logically ordered steps, labeled as [Step 1] → [Step 2] → ... → [Final Step].")
            elif decorator == '+++Socratic':
                decorator_instructions.append("You must use the Socratic method: [Restate Question] → [Clarify Definitions] → [Analyze Assumptions] → [Explore Perspectives] → [Use Analogies/Examples] → [Encourage Further Inquiry].")
            elif decorator == '+++Debate':
                decorator_instructions.append("You must analyze multiple viewpoints before reaching a conclusion: [State Position] → [Perspective 1] → [Perspective 2] → ... → [Analysis & Rebuttal] → [Conclusion].")
            elif decorator == '+++Critique':
                decorator_instructions.append("You must provide constructive criticism: [Identify Subject] → [Highlight Strengths] → [Critique Weaknesses] → [Suggest Improvements] → [Constructive Conclusion].")
            elif decorator.startswith('+++Refine'):
                iterations = re.search(r'iterations=(\d+)', decorator)
                if iterations:
                    n = int(iterations.group(1))
                    decorator_instructions.append(f"You must refine your answer {n} times: [Iteration 1] → [Iteration 2] → ... → [Final Answer].")
            elif decorator == '+++CiteSources':
                decorator_instructions.append("You must support all claims with credible references: [Initial Answer] → [Identify Key Claims] → [Find Credible Sources] → [Integrate Citations] → [Provide Full References] → [Verify Credibility] → [Final Answer].")
            elif decorator == '+++FactCheck':
                decorator_instructions.append("You must verify factual accuracy: [Initial Answer] → [Identify Claims] → [Research & Verify] → [Mark Uncertainties] → [Provide Verified Sources] → [Final Answer].")
            elif decorator.startswith('+++OutputFormat'):
                format_type = re.search(r'format=(\w+)', decorator)
                if format_type:
                    decorator_instructions.append(f"You must format your response in {format_type.group(1)}.")
            elif decorator.startswith('+++Tone'):
                style = re.search(r'style=(\w+)', decorator)
                if style:
                    decorator_instructions.append(f"You must use a {style.group(1)} tone in your response.")
        
        # Add all decorator instructions at the beginning of the prompt
        if decorator_instructions:
            enhanced_prompt = "IMPORTANT INSTRUCTIONS:\n" + "\n".join(decorator_instructions) + "\n\n" + enhanced_prompt
            
        return enhanced_prompt

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
    """PlanGEN implementation for GAIA dataset evaluation"""
    
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
        self.prompt_decorator = PromptDecorator()
        
    def load_dataset(self):
        """Load the GAIA dataset"""
        self.dataset = load_gaia_dataset()
        return self.dataset
    
    def _enhance_solution_prompt(self, problem: str, constraints: List[str], decorators: set[str]) -> str:
        """Enhance the solution generation prompt with decorators"""
        base_prompt = f"""Given the following problem:

{problem}

And considering these constraints:

{chr(10).join(f'- {c}' for c in constraints)}

Generate a detailed solution to the problem."""

        # Apply decorators to enhance the prompt
        enhanced_prompt = self.prompt_decorator.apply_decorators(base_prompt, decorators)
        return enhanced_prompt
        
    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve a problem using REBASE algorithm with prompt decorators"""
        # Extract constraints
        constraints = self.constraint_agent.run(problem)
        
        # Apply prompt decorators to the problem
        clean_problem, decorators = self.prompt_decorator.parse_decorators(problem)
        
        # Enhance the solution generation prompt
        enhanced_prompt = self._enhance_solution_prompt(clean_problem, constraints, decorators)
        
        # Generate solutions using REBASE with enhanced prompt
        best_plan, score, rebase_metadata = self.rebase.run(enhanced_prompt)
        
        # Verify the solution
        verification_result = self.verification_agent.run(enhanced_prompt, constraints, best_plan)
        
        # Select the best solution
        selected_solution = {
            'selected_solution': best_plan,
            'score': score,
            'verification_result': verification_result
        }
        
        return {
            'problem': problem,
            'constraints': constraints,
            'solutions': [best_plan],
            'verification_results': [verification_result],
            'selected_solution': selected_solution,
            'rebase_metadata': rebase_metadata,
            'applied_decorators': list(decorators),
            'enhanced_prompt': enhanced_prompt
        }
    
    def evaluate_dataset(self, num_samples: int = None) -> List[Dict[str, Any]]:
        """Evaluate PlanGEN on GAIA dataset using REBASE algorithm"""
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
    
    # Run evaluation using REBASE algorithm for first 5 samples
    results = plangen.evaluate_dataset(num_samples=2)
    
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
        console.print(f"Applied Decorators: {result.get('applied_decorators')}")
        console.print(f"Enhanced Prompt: {result.get('enhanced_prompt')}")
        if 'error' in result:
            console.print(f"[bold red]Error: {result['error']}[/bold red]")

if __name__ == "__main__":
    main() 