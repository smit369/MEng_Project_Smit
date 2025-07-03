"""
PlanGEN implementation for GAIA dataset evaluation using Best of N algorithm with prompt decorators and challenger integration
"""

import os
import time
from typing import Dict, Any, List, Optional
from datasets import load_dataset
from huggingface_hub import login
from rich.console import Console
import json
import re
import pandas as pd
from datetime import datetime
import concurrent.futures

# Import from plangen package
from plangen import PlanGEN
from plangen.agents import ConstraintAgent, SelectionAgent, SolutionAgent, VerificationAgent
from plangen.agents.verification_agent_robust import RobustVerificationAgent
from plangen.models import BaseModelInterface, OpenAIModelInterface
from plangen.prompts import PromptManager
from plangen.utils.llm_interface import LLMInterface
from plangen.algorithms.best_of_n import BestOfN
from plangen.verification.strategies.gaia_verifier_llm import GaiaVerifierLLM

console = Console()

# --- Challenger System ---
class Challenger:
    """Base class for all challengers."""
    def act_on(self, plan: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Modify or critique the plan. Returns dict with at least 'plan' key."""
        raise NotImplementedError

class PromptDecoratorChallenger(Challenger):
    """Challenger that applies prompt decorators to a plan."""
    def __init__(self, decorators: set, prompt_decorator):
        self.decorators = decorators
        self.prompt_decorator = prompt_decorator
    def act_on(self, plan: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # context may include problem, constraints, etc.
        # Ensure constraints are never None
        if 'constraints' in context and context['constraints'] is None:
            context['constraints'] = []
        
        enhanced_plan = self.prompt_decorator.apply_decorators(plan, self.decorators)
        return {
            'plan': enhanced_plan,
            'applied_decorators': list(self.decorators),
            'original_plan': plan
        }

def ChallengerFactory(challenger_type: str, decorators: set, prompt_decorator) -> Challenger:
    if challenger_type == 'prompt_decorator':
        return PromptDecoratorChallenger(decorators, prompt_decorator)
    raise ValueError(f"Unknown challenger type: {challenger_type}")

# --- Prompt Decorator ---
class PromptDecorator:
    """Handles prompt decorators for solution generation"""
    def __init__(self):
        self.active_decorators = set()
        self.chat_scope = False
    def parse_decorators(self, prompt: str) -> tuple[str, set[str]]:
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
        if not decorators:
            return prompt
        enhanced_prompt = prompt
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
        if decorator_instructions:
            enhanced_prompt = "IMPORTANT INSTRUCTIONS:\n" + "\n".join(decorator_instructions) + "\n\n" + enhanced_prompt
        return enhanced_prompt

# --- Dataset Loader ---
def load_gaia_dataset():
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    login(token=os.getenv('HF_TOKEN'))
    console.print("[bold blue]Loading GAIA dataset from Hugging Face...[/bold blue]")
    dataset = load_dataset("gaia-benchmark/GAIA", '2023_all', token=os.getenv('HF_TOKEN'), trust_remote_code=True)
    console.print("\n[bold yellow]Dataset structure:[/bold yellow]")
    console.print(dataset['validation'].features)
    return dataset['validation']

# --- Main PlanGEN Class ---
class PlanGENGaia(PlanGEN):
    """PlanGEN implementation for GAIA dataset evaluation with Best of N and challenger integration."""
    def __init__(self, model: Optional[BaseModelInterface] = None, prompt_manager: Optional[PromptManager] = None,
                 num_solutions: int = 3, model_name: str = "llama3:8b", temperature: float = 0.7,
                 n_plans: int = 5, sampling_strategy: str = "diverse", parallel: bool = True):
        if model is None:
            model = LLMInterface(model_name=model_name, temperature=temperature, model_type="ollama")
        
        # Create a robust verification agent with the new LLM-based GAIA verifier
        robust_verification_agent = RobustVerificationAgent(
            llm_interface=model,
            model_name=model_name,
            temperature=temperature,
            verifier=GaiaVerifierLLM(model_name=model_name, temperature=0.1)  # Use LLM-based verifier
        )
        
        super().__init__(model=model, prompt_manager=prompt_manager, num_solutions=num_solutions)
        
        # Replace the default verification agent with the robust one
        self.verification_agent = robust_verification_agent
        
        self.best_of_n = BestOfN(
            n_plans=n_plans,
            sampling_strategy=sampling_strategy,
            parallel=parallel,
            domain="gaia",
            llm_interface=model,
            constraint_agent=self.constraint_agent,
            verification_agent=self.verification_agent,
            temperature=temperature
        )
        self.dataset = None
        self.prompt_decorator = PromptDecorator()
        # For now, always use PromptDecoratorChallenger
        self.challenger_type = 'prompt_decorator'
    def load_dataset(self):
        self.dataset = load_gaia_dataset()
        return self.dataset
    def _enhance_solution_prompt(self, problem: str, constraints: List[str], decorators: set[str]) -> str:
        # Ensure constraints is never None
        if constraints is None:
            constraints = []
            
        base_prompt = f"""Given the following problem:\n\n{problem}\n\nAnd considering these constraints:\n\n{chr(10).join(f'- {c}' for c in constraints)}\n\nGenerate a detailed solution to the problem."""
        enhanced_prompt = self.prompt_decorator.apply_decorators(base_prompt, decorators)
        return enhanced_prompt
    def solve(self, problem: str) -> Dict[str, Any]:
        try:
            # [Step 1] Problem Statement → ConstraintAgent
            constraints = self.constraint_agent.run(problem)
            if constraints is None:
                constraints = []
                
            clean_problem, _ = self.prompt_decorator.parse_decorators(problem)
            
            # List of decorators to apply (can be extended or loaded from file)
            decorator_list = [
                '+++StepByStep',
                '+++Reasoning',
                '+++Socratic',
                '+++Debate',
                '+++Critique',
                '+++CiteSources',
                '+++FactCheck',
            ]
            
            n_plans = self.best_of_n.n_plans
            plans = []
            used_decorators = []
            
            # Store all decorators in a set
            all_decorators = set(decorator_list)
            
            # [Step 2] Sampling Strategy - Generate N initial plans
            initial_plans = []
            try:
                # Select the sampling function from the BestOfN instance
                if self.best_of_n.sampling_strategy == "diverse":
                    sample_fn = self.best_of_n._diverse_sampling
                elif self.best_of_n.sampling_strategy == "adaptive":
                    # NOTE: Adaptive sampling requires scored results from previous plans,
                    # which is not compatible with the challenger flow of scoring at the end.
                    # Falling back to basic sampling.
                    sample_fn = self.best_of_n._basic_sampling
                else:  # 'basic'
                    sample_fn = self.best_of_n._basic_sampling

                # Generate plans, respecting the parallel flag
                if self.best_of_n.parallel and self.best_of_n.sampling_strategy == 'basic':
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(sample_fn, clean_problem, constraints, []) for _ in range(self.best_of_n.n_plans)]
                        for future in concurrent.futures.as_completed(futures):
                            initial_plans.append(future.result())
                else:
                    # Execute sequentially for diverse/adaptive strategies or if parallel is False
                    # This is necessary because diverse sampling needs the output of the previous plan
                    results_for_sampling = []
                    for _ in range(self.best_of_n.n_plans):
                        plan = sample_fn(clean_problem, constraints, results_for_sampling)
                        initial_plans.append(plan)
                        # For diverse sampling, we need to provide the previous plans.
                        # We pass a dummy tuple to satisfy the original method's signature.
                        results_for_sampling.append((plan, 0.0, ""))
                
                # Store the generated plans for the next step
                for plan in initial_plans:
                    # Initialize with dummy score/feedback as they are not calculated yet
                    plans.append((plan, 0.0, "", clean_problem, constraints))
                
            except Exception as e:
                console.print(f"[bold red]Error generating plans: {str(e)}[/bold red]")
                
            # If no plans were generated, handle the error gracefully
            if not plans:
                return {
                    'problem': problem,
                    'constraints': constraints,
                    'solutions': [],
                    'verification_results': [],
                    'selected_solution': None,
                    'best_of_n_metadata': {'challenged_plans': [], 'best_idx': -1},
                    'applied_decorators': [],
                    'enhanced_prompts': [],
                    'error': "No plans could be generated"
                }
                
            # [Step 3] Apply each decorator to each plan individually
            challenged_plans = []
            all_enhanced_prompts = []

            # Outer loop: Iterate through each of the N initial plans
            for i, (original_plan, _, _, clean_problem_ctx, constraints_ctx) in enumerate(plans):
                # Inner loop: Iterate through each of the available decorators
                for decorator in all_decorators:
                    try:
                        # Create a challenger with only a SINGLE decorator
                        challenger = ChallengerFactory(self.challenger_type, {decorator}, self.prompt_decorator)
                        
                        # Apply the single decorator to the original plan
                        challenge_result = challenger.act_on(original_plan, {'problem': problem, 'constraints': constraints_ctx})
                        
                        enhanced_plan = challenge_result['plan']
                        applied_decorators = challenge_result['applied_decorators']
                        
                        # Store the enhanced prompt for display/logging
                        all_enhanced_prompts.append(enhanced_plan)
                        
                        # Add the newly challenged plan to our list for scoring
                        challenged_plans.append({
                            'original_plan_idx': i,
                            'original_plan': original_plan,
                            'challenged_plan': enhanced_plan,
                            'applied_decorators': applied_decorators,
                            'score': 0.0,  # Score to be calculated in the next step
                            'feedback': "", # Feedback to be populated in the next step
                            'decorator_used': decorator
                        })
                        
                    except Exception as e:
                        console.print(f"[bold red]Error applying decorator '{decorator}' to plan {i+1}: {str(e)}[/bold red]")
                        # Optionally, add a failed entry to keep track
                        challenged_plans.append({
                            'original_plan_idx': i,
                            'original_plan': original_plan,
                            'challenged_plan': original_plan, # Fallback to original
                            'applied_decorators': [],
                            'score': 0.0,
                            'feedback': f"Error: {e}",
                            'decorator_used': decorator,
                            'error': True
                        })

            # If no challenged plans exist, handle the error gracefully
            if not challenged_plans:
                return {
                    'problem': problem,
                    'constraints': constraints,
                    'solutions': [],
                    'verification_results': [],
                    'selected_solution': None,
                    'best_of_n_metadata': {'challenged_plans': [], 'best_idx': -1},
                    'applied_decorators': [],
                    'enhanced_prompts': [],
                    'error': "All plan challenges failed"
                }
                
            # [Step 4] Score the challenged plans
            for i, plan_data in enumerate(challenged_plans):
                # Re-verify the challenged plan if needed
                enhanced_plan = plan_data['challenged_plan']
                verification_result = self.verification_agent.run(clean_problem, constraints, enhanced_plan)
                score_tuple = verification_result
                
                # Update the score and feedback
                challenged_plans[i]['score'] = score_tuple[1]  # Updated score
                challenged_plans[i]['feedback'] = score_tuple[0]  # Updated feedback
                
            # [Step 5] Select Best Plan
            best_idx = max(range(len(challenged_plans)), key=lambda i: challenged_plans[i]['score'])
            best_plan = challenged_plans[best_idx]['challenged_plan']
            best_score = challenged_plans[best_idx]['score']
            best_feedback = challenged_plans[best_idx]['feedback']
            
            # Metadata for the best of N results
            best_of_n_metadata = {
                'challenged_plans': challenged_plans,
                'best_idx': best_idx
            }
            
            # Final verification of the best plan
            verification_result = self.verification_agent.run(clean_problem, constraints, best_plan)
            
            # [Step 6] Return plan, score, metadata
            selected_solution = {
                'selected_solution': best_plan,
                'score': best_score,
                'feedback': best_feedback,
                'verification_result': verification_result
            }
            
            return {
                'problem': problem,
                'constraints': constraints,
                'solutions': [best_plan],
                'verification_results': [verification_result],
                'selected_solution': selected_solution,
                'best_of_n_metadata': best_of_n_metadata,
                'applied_decorators': list(all_decorators),
                'enhanced_prompts': all_enhanced_prompts
            }
        except Exception as e:
            # Catch-all for any unexpected errors in the solve method
            console.print(f"[bold red]Unexpected error in solve method: {str(e)}[/bold red]")
            return {
                'problem': problem,
                'constraints': [],  # Use empty list for constraints
                'solutions': [],
                'verification_results': [],
                'selected_solution': None,
                'best_of_n_metadata': {'challenged_plans': [], 'best_idx': -1},
                'applied_decorators': [],
                'enhanced_prompts': [],
                'error': str(e)
            }
            
    def save_results(self, results: List[Dict[str, Any]], output_dir: str = "results") -> None:
        """Save evaluation results to both JSON and CSV formats."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        json_path = os.path.join(output_dir, f"plangen_gaia_bestofn_prompt_challenger_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"[bold green]Full results saved to: {json_path}[/bold green]")
        
        # Extract key metrics for CSV
        csv_data = []
        for result in results:
            if 'runtime_metadata' in result:
                continue
                
            row = {
                'problem': result.get('problem', ''),
                'constraints': str(result.get('constraints', [])),
                'selected_solution': result.get('selected_solution', {}).get('selected_solution', ''),
                'score': result.get('selected_solution', {}).get('score', 0),
                'feedback': result.get('selected_solution', {}).get('feedback', ''),
                'processing_time_minutes': result.get('processing_time_minutes', 0),
                'applied_decorators': str(result.get('applied_decorators', [])),
                'level': result.get('gaia_metadata', {}).get('level', ''),
                'task_id': result.get('gaia_metadata', {}).get('task_id', ''),
                'file_name': result.get('gaia_metadata', {}).get('file_name', ''),
                'error': result.get('error', '')
            }
            csv_data.append(row)
        
        # Save summary as CSV
        csv_path = os.path.join(output_dir, f"plangen_gaia_bestofn_prompt_challenger_results_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        console.print(f"[bold green]Summary results saved to: {csv_path}[/bold green]")

    def evaluate_dataset(self, num_samples: int = None) -> List[Dict[str, Any]]:
        if self.dataset is None:
            self.load_dataset()
        samples = self.dataset
        if num_samples is not None:
            samples = samples.select(range(min(num_samples, len(samples))))
        results = []
        total_start_time = time.time()
        for idx, sample in enumerate(samples):
            sample_start_time = time.time()
            console.print(f"\n[bold blue]Processing sample {idx + 1}/{len(samples)}[/bold blue]")
            try:
                problem = sample['Question']
                result = self.solve(problem)
                result['gaia_metadata'] = {
                    'level': sample.get('Level', ''),
                    'ground_truth': sample.get('Final answer', ''),
                    'task_id': sample.get('task_id', ''),
                    'file_name': sample.get('file_name', ''),
                    'file_path': sample.get('file_path', ''),
                    'annotator_metadata': sample.get('Annotator Metadata', {})
                }
                sample_time = (time.time() - sample_start_time) / 60
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
                
        # Calculate total time only for successful samples
        successful_samples = len([r for r in results if 'error' not in r])
        total_time = (time.time() - total_start_time) / 60
        
        console.print(f"\n[bold green]Total runtime: {total_time:.2f} minutes[/bold green]")
        
        # Add runtime metadata
        avg_time = total_time / len(samples) if samples else 0
        results.append({
            'runtime_metadata': {
                'total_runtime_minutes': total_time,
                'average_time_per_sample_minutes': avg_time,
                'total_samples_processed': len(samples),
                'successful_samples': successful_samples
            }
        })

        # Save results
        self.save_results(results)
        
        return results

def main():
    plangen = PlanGENGaia(
        model_name="llama3:8b",
        temperature=0.7,
        n_plans=5,
        sampling_strategy="diverse",
        parallel=True
    )
    results = plangen.evaluate_dataset(num_samples=1)
    #results = plangen.evaluate_dataset()
    
    for result in results:
        if 'runtime_metadata' in result:
            console.print("\n[bold yellow]Runtime Summary:[/bold yellow]")
            console.print(f"Total Runtime: {result['runtime_metadata']['total_runtime_minutes']:.2f} minutes")
            console.print(f"Average Time per Sample: {result['runtime_metadata']['average_time_per_sample_minutes']:.2f} minutes")
            console.print(f"Total Samples Processed: {result['runtime_metadata']['total_samples_processed']}")
            console.print(f"Successful Samples: {result['runtime_metadata']['successful_samples']}")
            continue
        console.print("\n[bold green]Results for sample:[/bold green]")
        console.print(f"Problem: {result['problem']}")
        console.print(f"Constraints: {result.get('constraints')}")
        console.print(f"Generated Solutions: {result.get('solutions')}")
        console.print(f"Verification Results: {result.get('verification_results')}")
        console.print(f"Selected Solution: {result.get('selected_solution')}")
        console.print(f"Best of N Metadata: {result.get('best_of_n_metadata')}")
        console.print(f"GAIA Metadata: {result.get('gaia_metadata')}")
        console.print(f"Applied Decorators: {result.get('applied_decorators')}")
        console.print(f"Enhanced Prompts: {result.get('enhanced_prompts')}")
        console.print(f"Processing Time: {result.get('processing_time_minutes', 'N/A'):.2f} minutes")
        if 'error' in result:
            console.print(f"[bold red]Error: {result['error']}[/bold red]")

if __name__ == "__main__":
    main() 