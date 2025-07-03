"""Extract problems and responses from GAIA baseline results."""

import json
from pathlib import Path
from rich.console import Console
from datetime import datetime

console = Console()

def extract_baseline_data(input_file: Path, output_file: Path):
    """Extract problems and responses from baseline results."""
    
    console.print(f"[bold blue]Loading baseline results from: {input_file}[/bold blue]")
    
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)
            console.print(f"[bold green]Successfully loaded JSON file with {len(results)} items[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error loading baseline results: {str(e)}[/bold red]")
        return
    
    extracted_data = []
    
    for i, item in enumerate(results):
        try:
            # Skip metadata entries
            if 'runtime_metadata' in item or 'accuracy_metrics' in item:
                continue
                
            # Skip error entries
            if 'error' in item:
                console.print(f"[bold red]Skipping item {i} due to error: {item['error']}[/bold red]")
                continue
            
            # Extract problem and response
            problem = item.get('question', '')
            response = item.get('response', '')
            
            if not problem or not response:
                console.print(f"[bold red]Warning: Missing problem or response for item {i}[/bold red]")
                continue
            
            # Add to extracted data
            extracted_data.append({
                'problem': problem,
                'response': response,
                'index': i
            })
            
            # Print first few items for verification
            if i < 3:
                console.print(f"\n[bold yellow]Item {i+1}:[/bold yellow]")
                console.print(f"Problem: {problem[:100]}...")
                console.print(f"Response: {response[:100]}...")
                
        except (KeyError, IndexError) as e:
            console.print(f"[bold red]Error processing item {i}: {e}[/bold red]")
            continue
    
    console.print(f"\n[bold green]Extracted {len(extracted_data)} problems and responses[/bold green]")
    
    # Save extracted data
    try:
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=2)
        console.print(f"[bold green]Extracted data saved to: {output_file}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error saving extracted data: {str(e)}[/bold red]")

def main():
    """Main function to extract baseline data."""
    
    # Find the baseline results file
    results_dir = Path("results")
    if not results_dir.exists():
        console.print("[bold red]Results directory not found![/bold red]")
        return
    
    # Look for baseline files
    baseline_files = list(results_dir.glob("gaia_baseline_llama3_8b_results_*.json"))
    if not baseline_files:
        console.print("[bold red]No baseline results files found![/bold red]")
        return
    
    # Use the most recent file
    input_file = max(baseline_files, key=lambda x: x.stat().st_mtime)
    
    # Create output filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = results_dir / f"extracted_baseline_problems_responses_{timestamp}.json"
    
    # Extract the data
    extract_baseline_data(input_file, output_file)

if __name__ == "__main__":
    main() 