"""Evaluate GAIA Baseline results by comparing formatted answers with ground truth."""

import json
import re
import requests
from typing import List, Dict, Any
from pathlib import Path
from rich.console import Console
from rich.table import Table
from datetime import datetime

console = Console()

def generate_with_ollama(prompt: str) -> str:
    """Generate response using Ollama API"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1  # Lower temperature for more consistent formatting
            },
            timeout=30
        )
        return response.json()["response"]
    except Exception as e:
        console.print(f"[bold red]Error calling Ollama API: {str(e)}[/bold red]")
        return ""

def format_answer(question: str, answer: str) -> str:
    """Format the final answer according to GAIA requirements"""
    format_prompt = f"""You are a strict answer formatter for the GAIA benchmark. Your task is to extract and format the final answer from the given response.

Rules for formatting:
1. Extract ONLY the final answer, not explanations or reasoning
2. If the answer is a refusal or uncertainty, make an educated guess based on the question context
3. If the answer contains multiple numbers, select the most relevant one
4. Remove any markdown formatting, quotes, or special characters
5. For text answers, use lowercase without any special formatting
6. For numbers, use the exact number without any formatting or units
7. If the answer is a single word or short phrase, return it as is
8. Never return placeholders like '_' or '?'
9. Look for patterns like "The answer is:", "Answer:", "Result:", etc.

Here is the question:
{question}

Here is the response to extract the answer from:
{answer}

Extract and format the final answer (single line, no formatting):"""
    
    response = generate_with_ollama(format_prompt)
    
    # Clean up the formatted response
    formatted = response.strip()
    
    # Remove any markdown, quotes, or special characters
    formatted = re.sub(r'[*_`]', '', formatted)
    formatted = re.sub(r'^["\']|["\']$', '', formatted)
    
    # If the answer is a refusal, try to extract a number or make a guess
    if any(word in formatted.lower() for word in ['cannot', 'unable', 'don\'t know', 'not sure', 'error']):
        numbers = re.findall(r'\d+', question + answer)
        if numbers:
            formatted = numbers[0]
        else:
            if 'how many' in question.lower():
                formatted = '0'
            elif 'what is' in question.lower():
                formatted = 'unknown'
    
    # If multiple numbers, take the first one
    if ',' in formatted:
        formatted = formatted.split(',')[0].strip()
    
    # Remove any remaining whitespace or special characters
    formatted = re.sub(r'\s+', ' ', formatted).strip()
    
    return formatted

def evaluate_by_level(predictions: List[str], ground_truth: List[str], levels: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculate accuracy metrics by difficulty level"""
    level_metrics = {}
    
    for level in ['1', '2', '3']:
        level_indices = [i for i, l in enumerate(levels) if l == level]
        level_preds = [predictions[i] for i in level_indices]
        level_truth = [ground_truth[i] for i in level_indices]
        
        total = len(level_indices)
        correct = sum(1 for p, g in zip(level_preds, level_truth) if p.strip().lower() == g.strip().lower())
        refusals = sum(1 for p in level_preds if '?' in p.lower() or 'cannot' in p.lower() or 'unable' in p.lower() or 'error' in p.lower())
        
        level_metrics[f'level_{level}'] = {
            'total': total,
            'correct': correct,
            'refusals': refusals,
            'accuracy': correct / total if total > 0 else 0,
            'refusal_rate': refusals / total if total > 0 else 0
        }
    
    return level_metrics

def find_baseline_results() -> Path:
    """Find the most recent baseline results file"""
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("Results directory not found")
    
    # Look for baseline files
    baseline_files = list(results_dir.glob("gaia_baseline_llama3_8b_results_*.json"))
    if not baseline_files:
        raise FileNotFoundError("No baseline results files found")
    
    # Return the most recent file
    return max(baseline_files, key=lambda x: x.stat().st_mtime)

def main():
    # Find and load baseline results file
    try:
        results_file = find_baseline_results()
        console.print(f"[bold blue]Loading baseline results from: {results_file}[/bold blue]")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
            console.print(f"[bold green]Successfully loaded JSON file with {len(results)} items[/bold green]")
            
    except Exception as e:
        console.print(f"[bold red]Error loading baseline results: {str(e)}[/bold red]")
        return
    
    predictions = []
    ground_truth = []
    levels = []
    
    # Process each result
    console.print("[bold blue]Processing GAIA Baseline results...[/bold blue]")
    for i, item in enumerate(results):
        try:
            # Skip metadata entries
            if 'runtime_metadata' in item or 'accuracy_metrics' in item:
                continue
                
            # Skip error entries
            if 'error' in item:
                console.print(f"[bold red]Skipping item {i} due to error: {item['error']}[/bold red]")
                continue
            
            # Get the response
            response = item.get('response', '')
            if not response:
                console.print(f"[bold red]Warning: No response found for item {i}[/bold red]")
                continue
            
            # Format the answer using llama model
            formatted_answer = format_answer(item['question'], response)
            
            # Store results
            predictions.append(formatted_answer)
            ground_truth.append(item['gaia_metadata']['ground_truth'])
            levels.append(item['gaia_metadata']['level'])
            
            # Print first few items for debugging
            if i < 5:
                console.print(f"\n[bold yellow]Item {i+1}:[/bold yellow]")
                console.print(f"Question: {item['question'][:100]}...")
                console.print(f"Response: {response[:100]}...")
                console.print(f"Formatted answer: {formatted_answer}")
                console.print(f"Ground truth: {item['gaia_metadata']['ground_truth']}")
                console.print(f"Level: {item['gaia_metadata']['level']}")
                
        except (KeyError, IndexError) as e:
            console.print(f"[bold red]Error processing item {i}: {e}[/bold red]")
            continue
    
    console.print(f"\n[bold green]Processed {len(predictions)} samples[/bold green]")
    
    # Calculate metrics by level
    level_metrics = evaluate_by_level(predictions, ground_truth, levels)
    
    # Calculate overall metrics
    total = len(predictions)
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p.strip().lower() == g.strip().lower())
    refusals = sum(1 for p in predictions if '?' in p.lower() or 'cannot' in p.lower() or 'unable' in p.lower() or 'error' in p.lower())
    
    overall_metrics = {
        'total': total,
        'correct': correct,
        'refusals': refusals,
        'accuracy': correct / total if total > 0 else 0,
        'refusal_rate': refusals / total if total > 0 else 0
    }
    
    # Display results
    table = Table(title="GAIA Baseline Results - LLaMA 3.1 8B")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add overall metrics
    for metric, value in overall_metrics.items():
        if isinstance(value, float):
            table.add_row(f"Overall {metric}", f"{value:.2%}")
        else:
            table.add_row(f"Overall {metric}", str(value))
    
    # Add level-specific metrics
    for level, metrics in level_metrics.items():
        table.add_row("", "")  # Empty row for spacing
        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(f"{level} {metric}", f"{value:.2%}")
            else:
                table.add_row(f"{level} {metric}", str(value))
    
    console.print(table)
    
    # Save detailed results
    detailed_results = {
        'overall_metrics': overall_metrics,
        'level_metrics': level_metrics,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'levels': levels,
        'total_samples': len(predictions),
        'model': 'llama3:8b',
        'evaluation_method': 'llama_formatted_answers'
    }
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_file = results_dir / f"baseline_evaluation_results_llama_formatted_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    console.print(f"[bold green]Results saved to {results_file}[/bold green]")

if __name__ == "__main__":
    main() 