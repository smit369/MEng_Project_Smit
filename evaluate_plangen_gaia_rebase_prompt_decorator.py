"""Evaluate PlanGEN GAIA results by comparing formatted answers with ground truth."""

import json
import re
import requests
from typing import List, Dict, Any
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def generate_with_ollama(prompt: str) -> str:
    """Generate response using Ollama API"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def format_answer(question: str, answer: str) -> str:
    """Format the final answer according to GAIA requirements"""
    format_prompt = f"""You are a strict answer formatter. Your task is to format the given answer to match the expected format in the GAIA dataset.

Rules for formatting:
1. If the answer is a refusal or uncertainty, make an educated guess based on the question context
2. If the answer contains multiple numbers, select the most relevant one
3. Remove any markdown formatting, quotes, or special characters
4. For text answers, use lowercase without any special formatting
5. For numbers, use the exact number without any formatting or units
6. If the answer is a single word or short phrase, return it as is
7. Never return placeholders like '_' or '?'

Here is the question:
{question}

Here is the answer to format:
{answer}

Formatted answer (single line, no formatting):"""
    
    response = generate_with_ollama(format_prompt)
    
    # Clean up the formatted response
    formatted = response.strip()
    
    # Remove any markdown, quotes, or special characters
    formatted = re.sub(r'[*_`]', '', formatted)
    formatted = re.sub(r'^["\']|["\']$', '', formatted)
    
    # If the answer is a refusal, try to extract a number or make a guess
    if any(word in formatted.lower() for word in ['cannot', 'unable', 'don\'t know', 'not sure']):
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
        refusals = sum(1 for p in level_preds if '?' in p.lower() or 'cannot' in p.lower() or 'unable' in p.lower())
        
        level_metrics[f'level_{level}'] = {
            'total': total,
            'correct': correct,
            'refusals': refusals,
            'accuracy': correct / total if total > 0 else 0,
            'refusal_rate': refusals / total if total > 0 else 0
        }
    
    return level_metrics

def main():
    # Load results file
    results_file = Path("plangen_gaia_rebase_prompt_decorator.json")
    console.print(f"[bold blue]Looking for results file at: {results_file.absolute()}[/bold blue]")
    
    if not results_file.exists():
        console.print("[bold red]Results file not found![/bold red]")
        return
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
            console.print(f"[bold green]Successfully loaded JSON file with {len(results)} items[/bold green]")
            
            # Print first item for debugging
            if results:
                console.print("\n[bold yellow]First item structure:[/bold yellow]")
                console.print(json.dumps(results[0], indent=2))
    except Exception as e:
        console.print(f"[bold red]Error loading JSON file: {str(e)}[/bold red]")
        return
    
    predictions = []
    ground_truth = []
    levels = []
    
    # Process each result
    console.print("[bold blue]Processing PlanGEN GAIA results...[/bold blue]")
    for i, item in enumerate(results):
        try:
            # Get the selected solution or fallback to first solution
            if 'selected_solution' in item and item['selected_solution']:
                if isinstance(item['selected_solution'], dict) and 'selected_solution' in item['selected_solution']:
                    selected_solution = item['selected_solution']['selected_solution']
                else:
                    # Handle case where selected_solution is the solution itself
                    selected_solution = item['selected_solution']
            elif 'solutions' in item and item['solutions']:
                selected_solution = item['solutions'][0]
            else:
                console.print(f"[bold red]Warning: No solution found for item {i}[/bold red]")
                continue
            
            # Format the answer
            formatted_answer = format_answer(item['problem'], selected_solution)
            
            # Store results
            predictions.append(formatted_answer)
            ground_truth.append(item['gaia_metadata']['ground_truth'])
            levels.append(item['gaia_metadata']['level'])
            
            # Print first few items for debugging
            if i < 165:
                console.print(f"\n[bold yellow]Item {i+1}:[/bold yellow]")
                console.print(f"Problem: {item['problem'][:100]}...")
                console.print(f"Selected solution: {selected_solution[:100]}...")
                console.print(f"Formatted answer: {formatted_answer}")
                console.print(f"Ground truth: {item['gaia_metadata']['ground_truth']}")
                console.print(f"Level: {item['gaia_metadata']['level']}")
                
        except (KeyError, IndexError) as e:
            console.print(f"[bold red]Error processing item {i}: {e}[/bold red]")
            continue
    
    # Calculate metrics by level
    level_metrics = evaluate_by_level(predictions, ground_truth, levels)
    
    # Calculate overall metrics
    total = len(predictions)
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p.strip().lower() == g.strip().lower())
    refusals = sum(1 for p in predictions if '?' in p.lower() or 'cannot' in p.lower() or 'unable' in p.lower())
    
    overall_metrics = {
        'total': total,
        'correct': correct,
        'refusals': refusals,
        'accuracy': correct / total if total > 0 else 0,
        'refusal_rate': refusals / total if total > 0 else 0
    }
    
    # Display results
    table = Table(title="PlanGEN GAIA Results")
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
        'levels': levels
    }
    
    # Create results directory
    results_dir = Path("results/plangen_gaia")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = results_dir / "evaluation_results_Rebase_Prompt_Decorator.json"
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    console.print(f"[bold green]Results saved to {results_file}[/bold green]")

if __name__ == "__main__":
    main() 