"""
PlanGEN implementation for GAIA dataset evaluation using Best of N algorithm with YAML-configured prompt decorators
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
from plangen.verification.strategies.gaia_verifier_yaml import GaiaVerifierYAML

# Import configuration loader
from config_loader import load_config

# Import the new prompt decorators framework
import prompt_decorators as pd
from prompt_decorators import (
    load_decorator_definitions,
    get_available_decorators,
    create_decorator_instance,
    apply_decorator,
    apply_dynamic_decorators,
    extract_decorators_from_text
)

console = Console()

# --- Challenger System ---
class Challenger:
    """Base class for all challengers."""
    def act_on(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Modify or critique the problem. Returns dict with at least 'problem' key."""
        raise NotImplementedError

class PromptDecoratorChallenger(Challenger):
    """Challenger that applies prompt decorators to a problem using YAML configuration."""
    def __init__(self, decorator_name: str, decorator_params: Dict[str, Any] = None):
        self.decorator_name = decorator_name
        self.decorator_params = decorator_params or {}
        
    def act_on(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Apply the decorator using the new framework FIRST
        enhanced_problem = apply_decorator(self.decorator_name, problem, **self.decorator_params)
        
        # THEN add custom instruction text to guide constraint and plan generation
        instruction_text = self._get_instruction_text(self.decorator_name)
        enhanced_problem = enhanced_problem + "\n\n" + instruction_text
        
        return {
            'problem': enhanced_problem,
            'applied_decorators': [self.decorator_name],
            'original_problem': problem,
            'decorator_params': self.decorator_params,
            'instruction_text': instruction_text
        }
    
    def _get_instruction_text(self, decorator_name: str) -> str:
        """Get instruction text for each decorator to guide constraint and plan generation."""
        instructions = {
            "Socratic": "Please consider this problem from multiple angles and ask yourself probing questions to identify key constraints and develop a comprehensive plan. What assumptions are you making? What alternative perspectives should be considered?",
            
            "Debate": "Approach this problem as if you're preparing for a debate. Consider opposing viewpoints and potential counterarguments. What constraints might others identify? How would you defend your approach?",
            
            "StepByStep": "Break down this problem systematically. Identify each step needed for constraint generation and plan development. What are the logical dependencies? What information is missing?",
            
            "Reasoning": "Apply rigorous logical reasoning to this problem. What are the fundamental constraints? What logical conclusions can you draw? How do these inform your planning approach?",
            
            "FirstPrinciples": "Start from first principles when analyzing this problem. What are the fundamental truths? What constraints emerge from basic laws or principles? How do these shape your planning strategy?",
            
            "TreeOfThought": "Explore multiple reasoning paths for this problem. Consider different branches of thought for constraint identification and plan development. What are the key decision points?",
            
            "RedTeam": "Act as a red team challenging your own approach to this problem. What constraints might you be missing? What weaknesses exist in your initial thinking? How can you strengthen your analysis?",
            
            "Abductive": "Use abductive reasoning to identify the most likely constraints and planning approach for this problem. What hypotheses explain the situation? What evidence supports each?",
            
            "Analogical": "Find analogies to help understand this problem's constraints and planning requirements. What similar situations can guide your thinking? What patterns apply here?",
            
            "BlindSpots": "Identify potential blind spots in your analysis of this problem. What constraints might you be overlooking? What perspectives are you missing? How can you address these gaps?",
            
            "Contrarian": "Take a contrarian view of this problem. What constraints might others ignore? What unconventional planning approaches could work? Challenge conventional wisdom.",
            
            "Deductive": "Use deductive reasoning to identify constraints and develop plans for this problem. What general principles apply? What specific conclusions can you draw?",
            
            "ForcedAnalogy": "Force analogies between this problem and unrelated domains to identify hidden constraints and planning insights. What can you learn from unexpected comparisons?",
            
            "Inductive": "Use inductive reasoning to identify patterns and constraints in this problem. What observations can you make? What generalizations apply? How do these inform planning?",
            
            "NegativeSpace": "Consider what's NOT being asked or addressed in this problem. What constraints might exist in the negative space? What planning elements are being overlooked?",
            
            "PerspectiveCascading": "Cascade through multiple perspectives when analyzing this problem. How do different viewpoints reveal different constraints? What planning insights emerge from perspective shifts?",
            
            "RootCause": "Identify the root causes underlying this problem. What fundamental constraints drive the situation? What planning approach addresses the core issues?",
            
            "TemporalReasoning": "Consider temporal aspects of this problem. What time-based constraints exist? How do past, present, and future considerations affect planning? What timing dependencies matter?"
        }
        return instructions.get(decorator_name, "Consider this problem carefully and identify key constraints and planning requirements.")

def ChallengerFactory(challenger_type: str, decorator_name: str, decorator_params: Dict[str, Any] = None) -> Challenger:
    if challenger_type == 'prompt_decorator':
        return PromptDecoratorChallenger(decorator_name, decorator_params)
    raise ValueError(f"Unknown challenger type: {challenger_type}")

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
    """PlanGEN implementation for GAIA dataset evaluation with YAML-configured prompt decorators."""
    def __init__(self, model: Optional[BaseModelInterface] = None, prompt_manager: Optional[PromptManager] = None,
                 num_solutions: int = 3, model_name: str = "llama3:8b", temperature: float = 0.7,
                 n_plans: int = 5, sampling_strategy: str = "diverse", parallel: bool = True):
        
        # Load configuration
        self.config = load_config()
        
        # Override parameters with config if not explicitly provided
        model_config = self.config.get_model_config()
        if model_name == "llama3:8b":
            model_name = model_config.name
        if temperature == 0.7:
            temperature = model_config.temperature
        best_of_n_config = self.config.get_best_of_n_config()
        if n_plans == 5:
            n_plans = best_of_n_config.n_plans
        if sampling_strategy == "diverse":
            sampling_strategy = best_of_n_config.sampling_strategy
        if parallel == True:
            parallel = best_of_n_config.parallel
        
        if model is None:
            model = LLMInterface(model_name=model_name, temperature=temperature, model_type="ollama")
        
        # Create a robust verification agent with the YAML-configured GAIA verifier
        robust_verification_agent = RobustVerificationAgent(
            llm_interface=model,
            model_name=model_name,
            temperature=temperature,
            verifier=GaiaVerifierYAML(model_name=model_name, temperature=0.1)  # Use YAML-configured verifier
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
        
        # Load the new prompt decorators framework
        try:
            load_decorator_definitions()
            console.print("[bold green]Loaded prompt decorators framework[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error loading prompt decorators framework: {e}[/bold red]")
        
        # For now, always use PromptDecoratorChallenger
        self.challenger_type = 'prompt_decorator'
        
    def load_dataset(self):
        self.dataset = load_gaia_dataset()
        return self.dataset
        
    def solve(self, problem: str) -> Dict[str, Any]:
        try:
            # [Step 1] Get ALL enabled challenger decorators (no adaptive selection)
            challenger_decorators = self.config.get_enabled_decorators()
            
            console.print(f"[bold blue]Using ALL challenger decorators: {challenger_decorators}[/bold blue]")
            
            # [Step 2] Apply ALL challenger decorators to enhance the problem BEFORE constraint generation
            enhanced_problems = []
            all_instruction_texts = []
            working_decorators = []
            failed_decorators = []
            
            for decorator_name in challenger_decorators:
                try:
                    # Get decorator configuration from YAML
                    decorator_config = self.config.get_decorator_params(decorator_name, 'challenger')
                    
                    # Provide sensible defaults for common parameters if they're missing
                    default_params = {
                        'Socratic': {'iterations': 3},
                        'Debate': {'perspectives': 3, 'balanced': True},
                        'StepByStep': {'numbered': True},
                        'FirstPrinciples': {'depth': 3},
                        'TreeOfThought': {'branches': 3, 'depth': 3, 'pruning': False},
                        'RedTeam': {'strength': 'moderate', 'focus': [], 'constructive': True},
                        'Abductive': {'hypotheses': 3, 'criteria': ['simplicity', 'explanatory_power'], 'rank': True},
                        'Analogical': {'domain': 'general', 'count': 2, 'depth': 'moderate'},
                        'BlindSpots': {'categories': ['cultural', 'temporal'], 'depth': 'thorough', 'position': 'after'},
                        'Contrarian': {'approach': 'devils-advocate', 'maintain': False, 'focus': ''},
                        'Deductive': {'premises': 2, 'formal': False, 'steps': 3},
                        'Inductive': {'examples': 5, 'confidence': True, 'structure': 'generalization'},
                        'ForcedAnalogy': {'source': 'biology', 'target': 'technology', 'strength': 'strong'},
                        'NegativeSpace': {'dimensions': 3, 'exploration_depth': 'comprehensive', 'counterfactual': True},
                        'PerspectiveCascading': {'levels': 3, 'cascade_type': 'hierarchical', 'integration': True},
                        'RootCause': {'levels': 3, 'analysis_type': 'systematic', 'solutions': True},
                        'TemporalReasoning': {'timeframes': ['past', 'present', 'future'], 'causality': True, 'prediction': True}
                    }
                    
                    # Merge default params with config params, with config taking precedence
                    if decorator_name in default_params:
                        merged_config = default_params[decorator_name].copy()
                        merged_config.update(decorator_config)
                        decorator_config = merged_config
                    
                    # Create a challenger with the new framework
                    challenger = ChallengerFactory(self.challenger_type, decorator_name, decorator_config)
                    
                    # Apply the decorator to enhance the problem
                    challenge_result = challenger.act_on(problem, {'problem': problem})
                    
                    enhanced_problem = challenge_result['problem']
                    instruction_text = challenge_result['instruction_text']
                    
                    # Store the enhanced problem and instruction text
                    enhanced_problems.append(enhanced_problem)
                    all_instruction_texts.append(instruction_text)
                    working_decorators.append(decorator_name)
                    
                    console.print(f"[bold green]Applied decorator '{decorator_name}' with instruction: {instruction_text[:100]}...[/bold green]")
                    
                except Exception as e:
                    console.print(f"[bold yellow]Warning: Decorator '{decorator_name}' failed: {str(e)} - Skipping this decorator[/bold yellow]")
                    # Skip this decorator but continue with others
                    failed_decorators.append(decorator_name)
                    # Don't add anything to enhanced_problems or all_instruction_texts for failed decorators
                    continue
            
            console.print(f"[bold blue]Successfully applied {len(working_decorators)} decorators, skipped {len(failed_decorators)} decorators[/bold blue]")
            
            # If no decorators worked, use the original problem
            if not enhanced_problems:
                console.print("[bold yellow]No decorators worked, using original problem[/bold yellow]")
                enhanced_problems = [problem]
                all_instruction_texts = [""]
                working_decorators = ["original"]
            
            # [Step 3] Generate constraints for each enhanced problem
            all_constraints = []
            for enhanced_problem in enhanced_problems:
                try:
                    constraints = self.constraint_agent.run(enhanced_problem)
                    if constraints is None:
                        constraints = []
                    all_constraints.append(constraints)
                except Exception as e:
                    console.print(f"[bold red]Error generating constraints: {str(e)}[/bold red]")
                    all_constraints.append([])
            
            # [Step 4] Early verification of ALL constraints (NEW)
            early_verification_results = []
            if self.config.get_early_verification_config()['enabled']:
                early_verification_decorators = self.config.get_early_verification_decorators()
                console.print(f"[bold blue]Applying early verification with decorators: {early_verification_decorators}[/bold blue]")
                
                for i, (enhanced_problem, constraints) in enumerate(zip(enhanced_problems, all_constraints)):
                    try:
                        # Apply early verification decorators to constraints
                        early_verified_constraints = self._apply_early_verification(enhanced_problem, constraints)
                        early_verification_results.append({
                            'problem_index': i,
                            'original_constraints': constraints,
                            'verified_constraints': early_verified_constraints,
                            'applied_decorators': early_verification_decorators
                        })
                        all_constraints[i] = early_verified_constraints  # Update with verified constraints
                        
                        console.print(f"[bold green]Early verification completed for problem {i+1}[/bold green]")
                        
                    except Exception as e:
                        console.print(f"[bold red]Error in early verification for problem {i+1}: {str(e)}[/bold red]")
                        early_verification_results.append({
                            'problem_index': i,
                            'original_constraints': constraints,
                            'verified_constraints': constraints,  # Keep original if verification fails
                            'applied_decorators': [],
                            'error': str(e)
                        })
            
            # [Step 5] Combine ALL constraints from all challengers for plan generation
            combined_constraints = []
            for constraints_list in all_constraints:
                combined_constraints.extend(constraints_list)
            
            # Remove duplicates while preserving order
            seen_constraints = set()
            unique_constraints = []
            for constraint in combined_constraints:
                if constraint not in seen_constraints:
                    seen_constraints.add(constraint)
                    unique_constraints.append(constraint)
            
            console.print(f"[bold blue]Combined {len(combined_constraints)} constraints into {len(unique_constraints)} unique constraints[/bold blue]")
            
            # [Step 6] Generate plans using Best of N - ONE PLAN PER CHALLENGER using combined constraints
            n_plans = len(working_decorators)  # One plan per working decorator
            plans = []
            
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

                # Generate ONE PLAN PER CHALLENGER using combined constraints
                # Each challenger gets its own plan but uses the comprehensive combined constraints
                if self.best_of_n.parallel and self.best_of_n.sampling_strategy == 'basic':
                    # Parallel generation: each challenger gets its own plan
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = []
                        for i, decorator_name in enumerate(working_decorators):
                            # Create a context for each challenger
                            challenger_context = {
                                'challenger_index': i,
                                'decorator_name': decorator_name,
                                'enhanced_problem': enhanced_problems[i],
                                'instruction_text': all_instruction_texts[i]
                            }
                            # Submit plan generation for this challenger
                            future = executor.submit(sample_fn, problem, unique_constraints, [], challenger_context)
                            futures.append((future, decorator_name))
                        
                        # Collect results
                        for future, decorator_name in concurrent.futures.as_completed(futures):
                            try:
                                plan = future.result()
                                plans.append({
                                    'plan': plan,
                                    'decorator_name': decorator_name,
                                    'challenger_index': futures.index((future, decorator_name))
                                })
                                console.print(f"[bold green]Generated plan for challenger '{decorator_name}'[/bold green]")
                            except Exception as e:
                                console.print(f"[bold red]Error generating plan for challenger '{decorator_name}': {str(e)}[/bold red]")
                                plans.append({
                                    'plan': f"Error generating plan: {str(e)}",
                                    'decorator_name': decorator_name,
                                    'challenger_index': futures.index((future, decorator_name))
                                })
                else:
                    # Sequential generation for diverse/adaptive strategies
                    results_for_sampling = []
                    for i, decorator_name in enumerate(working_decorators):
                        # Create a context for each challenger
                        challenger_context = {
                            'challenger_index': i,
                            'decorator_name': decorator_name,
                            'enhanced_problem': enhanced_problems[i],
                            'instruction_text': all_instruction_texts[i]
                        }
                        
                        # Generate plan for this challenger
                        plan = sample_fn(problem, unique_constraints, results_for_sampling, challenger_context)
                        plans.append({
                            'plan': plan,
                            'decorator_name': decorator_name,
                            'challenger_index': i
                        })
                        
                        # For diverse sampling, we need to provide the previous plans
                        # We pass a dummy tuple to satisfy the original method's signature
                        results_for_sampling.append((plan, 0.0, ""))
                        
                        console.print(f"[bold green]Generated plan for challenger '{decorator_name}'[/bold green]")
                
            except Exception as e:
                console.print(f"[bold red]Error generating plans: {str(e)}[/bold red]")
                
            # If no plans were generated, handle the error gracefully
            if not plans:
                return {
                    'problem': problem,
                    'enhanced_problems': enhanced_problems,
                    'all_constraints': all_constraints,
                    'combined_constraints': unique_constraints,
                    'early_verification_results': early_verification_results,
                    'solutions': [],
                    'verification_results': [],
                    'selected_solution': None,
                    'best_of_n_metadata': {'plans': [], 'best_idx': -1},
                    'applied_decorators': working_decorators,
                    'failed_decorators': failed_decorators,
                    'instruction_texts': all_instruction_texts,
                    'error': "No plans could be generated"
                }
                
            # [Step 7] Score the generated plans
            scored_plans = []
            for i, plan_data in enumerate(plans):
                try:
                    # Use the original problem and combined constraints for verification
                    verification_result = self.verification_agent.run(problem, unique_constraints, plan_data['plan'])
                    score_tuple = verification_result
                    
                    scored_plans.append({
                        'plan': plan_data['plan'],
                        'score': score_tuple[1],
                        'feedback': score_tuple[0],
                        'plan_index': i,
                        'decorator_name': plan_data['decorator_name'],
                        'challenger_index': plan_data['challenger_index']
                    })
                    
                except Exception as e:
                    console.print(f"[bold red]Error scoring plan {i+1} for decorator '{plan_data['decorator_name']}': {str(e)}[/bold red]")
                    scored_plans.append({
                        'plan': plan_data['plan'],
                        'score': 0.0,
                        'feedback': f"Error: {e}",
                        'plan_index': i,
                        'decorator_name': plan_data['decorator_name'],
                        'challenger_index': plan_data['challenger_index']
                    })
            
            # [Step 8] Select Best Plan
            if scored_plans:
                best_idx = max(range(len(scored_plans)), key=lambda i: scored_plans[i]['score'])
                best_plan = scored_plans[best_idx]['plan']
                best_score = scored_plans[best_idx]['score']
                best_feedback = scored_plans[best_idx]['feedback']
            else:
                best_plan = ""
                best_score = 0.0
                best_feedback = "No plans available"
                best_idx = -1
            
            # Metadata for the best of N results
            best_of_n_metadata = {
                'plans': scored_plans,
                'best_idx': best_idx
            }
            
            # Final verification of the best plan
            if best_plan:
                verification_result = self.verification_agent.run(problem, unique_constraints, best_plan)
            else:
                verification_result = ("No plan to verify", 0.0)
            
            # [Step 9] Return plan, score, metadata
            selected_solution = {
                'selected_solution': best_plan,
                'score': best_score,
                'feedback': best_feedback,
                'verification_result': verification_result
            }
            
            return {
                'problem': problem,
                'enhanced_problems': enhanced_problems,
                'all_constraints': all_constraints,
                'combined_constraints': unique_constraints,
                'early_verification_results': early_verification_results,
                'solutions': [best_plan],
                'verification_results': [verification_result],
                'selected_solution': selected_solution,
                'best_of_n_metadata': best_of_n_metadata,
                'applied_decorators': working_decorators,
                'failed_decorators': failed_decorators,
                'instruction_texts': all_instruction_texts,
                'config_source': self.config.config_path
            }
        except Exception as e:
            # Catch-all for any unexpected errors in the solve method
            console.print(f"[bold red]Unexpected error in solve method: {str(e)}[/bold red]")
            return {
                'problem': problem,
                'enhanced_problems': [],
                'all_constraints': [],
                'combined_constraints': [],
                'early_verification_results': [],
                'solutions': [],
                'verification_results': [],
                'selected_solution': None,
                'best_of_n_metadata': {'plans': [], 'best_idx': -1},
                'applied_decorators': [],
                'failed_decorators': [],
                'instruction_texts': [],
                'error': str(e)
            }
            
    def _apply_early_verification(self, enhanced_problem: str, constraints: List[str]) -> List[str]:
        """Apply early verification decorators to constraints."""
        try:
            early_verification_decorators = self.config.get_early_verification_decorators()
            verified_constraints = constraints.copy()
            
            for decorator_name in early_verification_decorators:
                try:
                    # Get early verification decorator configuration
                    decorator_config = self.config.get_early_verification_decorator_config(decorator_name)
                    
                    # Apply decorator to each constraint
                    for i, constraint in enumerate(verified_constraints):
                        enhanced_constraint = apply_decorator(decorator_name, constraint, **decorator_config)
                        verified_constraints[i] = enhanced_constraint
                    
                    console.print(f"[bold green]Applied early verification decorator '{decorator_name}' to constraints[/bold green]")
                    
                except Exception as e:
                    console.print(f"[bold red]Error applying early verification decorator '{decorator_name}': {str(e)}[/bold red]")
                    continue
            
            return verified_constraints
            
        except Exception as e:
            console.print(f"[bold red]Error in early verification: {str(e)}[/bold red]")
            return constraints  # Return original constraints if verification fails

    def save_results(self, results: List[Dict[str, Any]], output_dir: str = "results") -> None:
        """Save evaluation results to both JSON and CSV formats."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        json_path = os.path.join(output_dir, f"plangen_gaia_yaml_configured_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"[bold green]Full results saved to: {json_path}[/bold green]")
        
        # Extract key metrics for CSV
        csv_data = []
        for result in results:
            if 'runtime_metadata' in result:
                continue
                
            selected_solution = result.get('selected_solution') or {}
            
            # Get best plan decorator information
            best_plan_decorator = "N/A"
            best_of_n_metadata = result.get('best_of_n_metadata', {})
            if best_of_n_metadata and 'plans' in best_of_n_metadata and best_of_n_metadata['best_idx'] >= 0:
                best_plan = best_of_n_metadata['plans'][best_of_n_metadata['best_idx']]
                best_plan_decorator = best_plan.get('decorator_name', 'N/A')
            
            row = {
                'problem': result.get('problem', ''),
                'enhanced_problems_count': len(result.get('enhanced_problems', [])),
                'constraints_count': len(result.get('all_constraints', [])),
                'combined_constraints_count': len(result.get('combined_constraints', [])),
                'early_verification_enabled': len(result.get('early_verification_results', [])) > 0,
                'early_verification_decorators': str([r.get('applied_decorators', []) for r in result.get('early_verification_results', [])]),
                'selected_solution': selected_solution.get('selected_solution', ''),
                'score': selected_solution.get('score', 0),
                'feedback': selected_solution.get('feedback', ''),
                'best_plan_decorator': best_plan_decorator,
                'working_decorators_count': len(result.get('applied_decorators', [])),
                'failed_decorators_count': len(result.get('failed_decorators', [])),
                'failed_decorators': str(result.get('failed_decorators', [])),
                'processing_time_minutes': result.get('processing_time_minutes', 0),
                'applied_decorators': str(result.get('applied_decorators', [])),
                'instruction_texts': str(result.get('instruction_texts', [])),
                'level': result.get('gaia_metadata', {}).get('level', ''),
                'task_id': result.get('gaia_metadata', {}).get('task_id', ''),
                'file_name': result.get('gaia_metadata', {}).get('file_name', ''),
                'error': result.get('error', ''),
                'config_source': result.get('config_source', '')
            }
            csv_data.append(row)
        
        # Save summary as CSV
        csv_path = os.path.join(output_dir, f"plangen_gaia_yaml_configured_results_{timestamp}.csv")
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
                'successful_samples': successful_samples,
                'config_source': self.config.config_path
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
    # Run on 10 samples for testing
    results = plangen.evaluate_dataset(num_samples=1)
    # Uncomment the line below to run on all samples
    # results = plangen.evaluate_dataset()
    
    for result in results:
        if 'runtime_metadata' in result:
            console.print("\n[bold yellow]Runtime Summary:[/bold yellow]")
            console.print(f"Total Runtime: {result['runtime_metadata']['total_runtime_minutes']:.2f} minutes")
            console.print(f"Average Time per Sample: {result['runtime_metadata']['average_time_per_sample_minutes']:.2f} minutes")
            console.print(f"Total Samples Processed: {result['runtime_metadata']['total_samples_processed']}")
            console.print(f"Successful Samples: {result['runtime_metadata']['successful_samples']}")
            console.print(f"Config Source: {result['runtime_metadata']['config_source']}")
            continue
        console.print("\n[bold green]Results for sample:[/bold green]")
        console.print(f"Problem: {result['problem']}")
        console.print(f"Enhanced Problems Count: {len(result.get('enhanced_problems', []))}")
        console.print(f"Constraints Count: {len(result.get('all_constraints', []))}")
        console.print(f"Combined Constraints Count: {len(result.get('combined_constraints', []))}")
        console.print(f"Working Decorators: {len(result.get('applied_decorators', []))} - {result.get('applied_decorators', [])}")
        console.print(f"Failed Decorators: {len(result.get('failed_decorators', []))} - {result.get('failed_decorators', [])}")
        console.print(f"Early Verification Enabled: {len(result.get('early_verification_results', [])) > 0}")
        if result.get('early_verification_results'):
            console.print(f"Early Verification Results: {len(result.get('early_verification_results', []))} problems verified")
            for i, ev_result in enumerate(result.get('early_verification_results', [])):
                console.print(f"  Problem {i+1}: {len(ev_result.get('applied_decorators', []))} decorators applied")
        console.print(f"Generated Solutions: {result.get('solutions')}")
        console.print(f"Verification Results: {result.get('verification_results')}")
        console.print(f"Selected Solution: {result.get('selected_solution')}")
        
        # Show best of N metadata with decorator information
        best_of_n_metadata = result.get('best_of_n_metadata', {})
        if best_of_n_metadata and 'plans' in best_of_n_metadata:
            console.print(f"Best of N Plans: {len(best_of_n_metadata['plans'])} plans generated")
            if best_of_n_metadata['plans']:
                best_plan = best_of_n_metadata['plans'][best_of_n_metadata['best_idx']]
                console.print(f"Best Plan Score: {best_plan.get('score', 'N/A')}")
                console.print(f"Best Plan Decorator: {best_plan.get('decorator_name', 'N/A')}")
        
        console.print(f"GAIA Metadata: {result.get('gaia_metadata')}")
        console.print(f"Instruction Texts: {result.get('instruction_texts')}")
        console.print(f"Processing Time: {result.get('processing_time_minutes', 'N/A'):.2f} minutes")
        console.print(f"Config Source: {result.get('config_source', 'N/A')}")
        if 'error' in result:
            console.print(f"[bold red]Error: {result['error']}[/bold red]")

if __name__ == "__main__":
    main() 