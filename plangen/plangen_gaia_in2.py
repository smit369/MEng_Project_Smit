"""
PlanGEN implementation for GAIA dataset evaluation
"""

import os
from typing import Dict, Any, List, Optional
from datasets import load_dataset
from huggingface_hub import login
from rich.console import Console
from dotenv import load_dotenv

# Import from plangen package
from plangen import PlanGEN
from plangen.agents import ConstraintAgent, SelectionAgent, SolutionAgent, VerificationAgent
from plangen.models import BaseModelInterface, OpenAIModelInterface
from plangen.prompts import PromptManager
from plangen.utils.llm_interface import LLMInterface

console = Console()

def load_gaia_dataset():
    """Load the GAIA dataset with authentication"""
    # Load environment variables
    load_dotenv()
    
    # Get Hugging Face token from environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set. Please set it in .env file or environment.")
    
    try:
        login(token=hf_token)
    except Exception as e:
        raise RuntimeError(f"Failed to login to Hugging Face: {str(e)}")
    
    console.print("[bold blue]Loading GAIA dataset from Hugging Face...[/bold blue]")
    try:
        dataset = load_dataset("gaia-benchmark/GAIA", '2023_all', token=hf_token, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load GAIA dataset: {str(e)}")
    
    # Print dataset structure for debugging
    console.print("\n[bold yellow]Dataset structure:[/bold yellow]")
    console.print(dataset['validation'].features)
    
    return dataset['validation']  # Using validation set for evaluation

// ... existing code ...

def main():
    """Main entry point for GAIA evaluation"""
    try:
        # Initialize PlanGEN with Llama model
        plangen = PlanGENGaia(
            model_name="llama2:8b",  # Updated model name
            temperature=0.7
        )
        
        # Run evaluation using complete PlanGEN workflow
        results = plangen.evaluate_dataset(num_samples=2)  # Start with 2 samples
        
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
    except Exception as e:
        console.print(f"[bold red]Fatal error: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()