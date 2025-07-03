"""
GAIA Baseline Model using llama3:8b
Simple baseline evaluation of GAIA dataset using direct llama3:8b model calls
"""

import os
import time
import json
import requests
from typing import Dict, Any, List
from datasets import load_dataset
from huggingface_hub import login
from rich.console import Console
from datetime import datetime

console = Console()

class GaiaBaselineLLaMA:
    """Simple baseline model for GAIA dataset evaluation using llama3:8b."""
    
    def __init__(self, model_name: str = "llama3:8b", temperature: float = 0.7):
        """Initialize the baseline model."""
        self.model_name = model_name
        self.temperature = temperature
        self.api_url = "http://localhost:11434/api/generate"
        
    def call_llm(self, prompt: str) -> str:
        """Call the llama3:8b model via Ollama API."""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            console.print(f"[bold red]Error calling LLM: {str(e)}[/bold red]")
            return f"Error: {str(e)}"
    
    def create_prompt(self, question: str) -> str:
        """Create a simple prompt for the GAIA question."""
        prompt = f"""You are an AI assistant helping to answer questions from the GAIA benchmark. 
Please provide a clear, accurate, and well-reasoned answer to the following question.

Question: {question}

Please provide your answer:"""
        return prompt
    
    def solve_problem(self, question: str) -> Dict[str, Any]:
        """Solve a single GAIA problem."""
        try:
            # Create prompt
            prompt = self.create_prompt(question)
            
            # Get response from LLM
            start_time = time.time()
            response = self.call_llm(prompt)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            return {
                'question': question,
                'response': response,
                'processing_time_seconds': processing_time,
                'model_name': self.model_name,
                'temperature': self.temperature
            }
            
        except Exception as e:
            console.print(f"[bold red]Error solving problem: {str(e)}[/bold red]")
            return {
                'question': question,
                'response': f"Error: {str(e)}",
                'processing_time_seconds': 0,
                'model_name': self.model_name,
                'temperature': self.temperature,
                'error': str(e)
            }

def load_gaia_dataset():
    """Load the GAIA dataset from Hugging Face."""
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    login(token=os.getenv('HF_TOKEN'))
    console.print("[bold blue]Loading GAIA dataset from Hugging Face...[/bold blue]")
    dataset = load_dataset("gaia-benchmark/GAIA", '2023_all', token=os.getenv('HF_TOKEN'), trust_remote_code=True)
    console.print("\n[bold yellow]Dataset structure:[/bold yellow]")
    console.print(dataset['validation'].features)
    return dataset['validation']

def evaluate_dataset(model: GaiaBaselineLLaMA, num_samples: int = None) -> List[Dict[str, Any]]:
    """Evaluate the GAIA dataset using the baseline model."""
    # Load dataset
    dataset = load_gaia_dataset()
    
    # Select samples
    if num_samples is not None:
        samples = dataset.select(range(min(num_samples, len(dataset))))
    else:
        samples = dataset
    
    console.print(f"[bold blue]Evaluating {len(samples)} samples...[/bold blue]")
    
    results = []
    total_start_time = time.time()
    
    for idx, sample in enumerate(samples):
        sample_start_time = time.time()
        console.print(f"\n[bold blue]Processing sample {idx + 1}/{len(samples)}[/bold blue]")
        
        try:
            question = sample['Question']
            
            # Solve the problem
            result = model.solve_problem(question)
            
            # Add GAIA metadata
            result['gaia_metadata'] = {
                'level': sample.get('Level', ''),
                'ground_truth': sample.get('Final answer', ''),
                'task_id': sample.get('task_id', ''),
                'file_name': sample.get('file_name', ''),
                'file_path': sample.get('file_path', ''),
                'annotator_metadata': sample.get('Annotator Metadata', {})
            }
            
            # Add processing time
            sample_time = (time.time() - sample_start_time)
            result['processing_time_seconds'] = sample_time
            
            console.print(f"[bold green]Sample processing time: {sample_time:.2f} seconds[/bold green]")
            console.print(f"[bold green]Response: {result['response'][:200]}...[/bold green]")
            
            results.append(result)
            
        except Exception as e:
            console.print(f"[bold red]Error processing sample {idx + 1}: {str(e)}[/bold red]")
            results.append({
                'error': str(e),
                'question': question if 'question' in locals() else None,
                'processing_time_seconds': (time.time() - sample_start_time),
                'model_name': model.model_name,
                'temperature': model.temperature
            })
    
    # Calculate total time
    total_time = time.time() - total_start_time
    successful_samples = len([r for r in results if 'error' not in r])
    
    console.print(f"\n[bold green]Total runtime: {total_time:.2f} seconds[/bold green]")
    console.print(f"[bold green]Successful samples: {successful_samples}/{len(samples)}[/bold green]")
    
    # Add runtime metadata
    avg_time = total_time / len(samples) if samples else 0
    results.append({
        'runtime_metadata': {
            'total_runtime_seconds': total_time,
            'average_time_per_sample_seconds': avg_time,
            'total_samples_processed': len(samples),
            'successful_samples': successful_samples,
            'model_name': model.model_name,
            'temperature': model.temperature
        }
    })
    
    return results

def calculate_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy metrics by comparing responses with ground truths."""
    total_samples = 0
    exact_matches = 0
    partial_matches = 0
    no_ground_truth = 0
    
    for result in results:
        if 'runtime_metadata' in result or 'error' in result:
            continue
            
        total_samples += 1
        ground_truth = result.get('gaia_metadata', {}).get('ground_truth', '')
        response = result.get('response', '')
        
        if not ground_truth:
            no_ground_truth += 1
            continue
            
        # Simple exact match comparison
        if response.strip().lower() == ground_truth.strip().lower():
            exact_matches += 1
        # Simple partial match (check if ground truth is contained in response)
        elif ground_truth.strip().lower() in response.strip().lower():
            partial_matches += 1
    
    accuracy_metrics = {
        'total_samples': total_samples,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'no_ground_truth': no_ground_truth,
        'exact_accuracy': (exact_matches / total_samples * 100) if total_samples > 0 else 0,
        'partial_accuracy': ((exact_matches + partial_matches) / total_samples * 100) if total_samples > 0 else 0
    }
    
    return accuracy_metrics

def save_results(results: List[Dict[str, Any]], output_dir: str = "results") -> None:
    """Save evaluation results to JSON file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy(results)
    
    # Add accuracy metrics to results
    results.append({
        'accuracy_metrics': accuracy_metrics
    })
    
    # Save results as JSON
    json_path = os.path.join(output_dir, f"gaia_baseline_llama3_8b_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"[bold green]Results saved to: {json_path}[/bold green]")
    
    # Print summary
    if results and 'runtime_metadata' in results[-2]:  # -2 because we added accuracy_metrics
        metadata = results[-2]['runtime_metadata']
        console.print(f"\n[bold yellow]Evaluation Summary:[/bold yellow]")
        console.print(f"Model: {metadata['model_name']}")
        console.print(f"Temperature: {metadata['temperature']}")
        console.print(f"Total Runtime: {metadata['total_runtime_seconds']:.2f} seconds")
        console.print(f"Average Time per Sample: {metadata['average_time_per_sample_seconds']:.2f} seconds")
        console.print(f"Total Samples Processed: {metadata['total_samples_processed']}")
        console.print(f"Successful Samples: {metadata['successful_samples']}")
        
        # Print accuracy metrics
        console.print(f"\n[bold yellow]Accuracy Metrics:[/bold yellow]")
        console.print(f"Total Samples: {accuracy_metrics['total_samples']}")
        console.print(f"Exact Matches: {accuracy_metrics['exact_matches']}")
        console.print(f"Partial Matches: {accuracy_metrics['partial_matches']}")
        console.print(f"No Ground Truth: {accuracy_metrics['no_ground_truth']}")
        console.print(f"Exact Accuracy: {accuracy_metrics['exact_accuracy']:.2f}%")
        console.print(f"Partial Accuracy: {accuracy_metrics['partial_accuracy']:.2f}%")

def main():
    """Main function to run the baseline evaluation."""
    console.print("[bold blue]GAIA Baseline Evaluation with llama3:8b[/bold blue]")
    console.print("=" * 60)
    
    # Initialize model
    model = GaiaBaselineLLaMA(
        model_name="llama3:8b",
        temperature=0.7
    )
    
    # Run evaluation on full dataset
    results = evaluate_dataset(model, num_samples=None)  # Evaluate all samples
    
    # Save results
    save_results(results)
    
    # Print sample results (first few for preview)
    console.print("\n[bold yellow]Sample Results (first 3):[/bold yellow]")
    sample_count = 0
    for result in results:
        if 'runtime_metadata' in result:
            continue
        if sample_count >= 3:  # Only show first 3 samples
            break
        console.print(f"\nQuestion: {result['question'][:100]}...")
        console.print(f"Response: {result['response'][:200]}...")
        console.print(f"Ground Truth: {result.get('gaia_metadata', {}).get('ground_truth', 'N/A')}")
        console.print(f"Processing Time: {result['processing_time_seconds']:.2f} seconds")
        if 'error' in result:
            console.print(f"[bold red]Error: {result['error']}[/bold red]")
        sample_count += 1

if __name__ == "__main__":
    main() 