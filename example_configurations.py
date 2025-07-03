#!/usr/bin/env python3
"""
Example configurations for PlanGEN GAIA System
Demonstrates how to use different configuration settings for various use cases.
"""

import yaml
from config_loader import PlanGENConfig, create_default_config
from rich.console import Console

console = Console()

def create_fast_test_config():
    """Create a configuration optimized for fast testing."""
    config = {
        'plangen': {
            'model': {
                'name': 'llama3:8b',
                'temperature': 0.3,  # Lower temperature for faster, more focused responses
                'model_type': 'ollama'
            },
            'best_of_n': {
                'n_plans': 3,  # Fewer plans for faster execution
                'sampling_strategy': 'basic',  # Basic strategy is faster
                'parallel': True,
                'num_solutions': 1
            },
            'evaluation': {
                'num_samples': 1,  # Test with just 1 sample
                'output_dir': 'results_fast_test',
                'save_formats': ['json']
            }
        },
        'prompt_decorators': {
            'enabled_decorators': ['StepByStep', 'Reasoning'],  # Only essential decorators
            'decorator_configs': {
                'StepByStep': {'numbered': True},
                'Reasoning': {'depth': 'basic'}  # Basic depth for speed
            }
        },
        'adaptive_config': {
            'enabled': False  # Disable adaptive config for simplicity
        },
        'performance': {
            'parallel': {'enabled': True, 'max_workers': 2},
            'timeouts': {'plan_generation': 60, 'verification': 30, 'total_per_sample': 300}
        }
    }
    
    with open('config_fast_test.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    console.print("[bold green]✓ Fast test configuration created: config_fast_test.yaml[/bold green]")

def create_research_config():
    """Create a configuration optimized for research and analysis."""
    config = {
        'plangen': {
            'model': {
                'name': 'llama3:8b',
                'temperature': 0.7,  # Balanced creativity
                'model_type': 'ollama'
            },
            'best_of_n': {
                'n_plans': 10,  # More plans for better diversity
                'sampling_strategy': 'diverse',  # Diverse strategy for research
                'parallel': True,
                'num_solutions': 3
            },
            'evaluation': {
                'num_samples': None,  # Process all samples
                'output_dir': 'results_research',
                'save_formats': ['json', 'csv']
            }
        },
        'prompt_decorators': {
            'enabled_decorators': ['StepByStep', 'Reasoning', 'Socratic', 'Debate', 'PeerReview', 'CiteSources', 'FactCheck'],
            'decorator_configs': {
                'StepByStep': {'numbered': True},
                'Reasoning': {'depth': 'comprehensive'},
                'Socratic': {'iterations': 5},
                'Debate': {'perspectives': 4, 'balanced': True},
                'PeerReview': {'criteria': ['accuracy', 'clarity', 'completeness'], 'style': 'constructive'},
                'CiteSources': {'style': 'inline', 'format': 'APA', 'comprehensive': True},
                'FactCheck': {'confidence': 0.9, 'strictness': 'high'}
            }
        },
        'adaptive_config': {
            'enabled': True,
            'problem_types': {
                'math': {
                    'decorators': ['StepByStep', 'Reasoning', 'FactCheck'],
                    'reasoning_depth': 'comprehensive',
                    'fact_check_strictness': 'high'
                },
                'debate': {
                    'decorators': ['Debate', 'PeerReview', 'CiteSources'],
                    'debate_perspectives': 5,
                    'citation_comprehensive': True
                },
                'analysis': {
                    'decorators': ['Socratic', 'Reasoning', 'FactCheck'],
                    'socratic_iterations': 5,
                    'reasoning_depth': 'comprehensive'
                }
            }
        },
        'performance': {
            'parallel': {'enabled': True, 'max_workers': 8},
            'timeouts': {'plan_generation': 600, 'verification': 180, 'total_per_sample': 3600}
        },
        'logging': {
            'level': 'DEBUG',
            'detailed': {
                'log_decorator_applications': True,
                'log_verification_scores': True,
                'log_plan_generation': True,
                'log_timing': True
            }
        }
    }
    
    with open('config_research.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    console.print("[bold green]✓ Research configuration created: config_research.yaml[/bold green]")

def create_creative_config():
    """Create a configuration optimized for creative problem solving."""
    config = {
        'plangen': {
            'model': {
                'name': 'llama3:8b',
                'temperature': 0.9,  # High creativity
                'model_type': 'ollama'
            },
            'best_of_n': {
                'n_plans': 8,  # More plans for creative diversity
                'sampling_strategy': 'diverse',
                'parallel': True,
                'num_solutions': 3
            },
            'evaluation': {
                'num_samples': 5,  # Test with a few samples
                'output_dir': 'results_creative',
                'save_formats': ['json', 'csv']
            }
        },
        'prompt_decorators': {
            'enabled_decorators': ['StepByStep', 'Reasoning', 'Socratic', 'Debate'],
            'decorator_configs': {
                'StepByStep': {'numbered': False},  # Less structured for creativity
                'Reasoning': {'depth': 'moderate'},
                'Socratic': {'iterations': 4},
                'Debate': {'perspectives': 5, 'balanced': False}  # Allow unbalanced perspectives for creativity
            }
        },
        'adaptive_config': {
            'enabled': True,
            'problem_types': {
                'creative': {
                    'decorators': ['StepByStep', 'Reasoning'],
                    'reasoning_depth': 'moderate',
                    'temperature': 0.9
                }
            }
        }
    }
    
    with open('config_creative.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    console.print("[bold green]✓ Creative configuration created: config_creative.yaml[/bold green]")

def create_academic_config():
    """Create a configuration optimized for academic rigor."""
    config = {
        'plangen': {
            'model': {
                'name': 'llama3:8b',
                'temperature': 0.5,  # Balanced for academic work
                'model_type': 'ollama'
            },
            'best_of_n': {
                'n_plans': 6,
                'sampling_strategy': 'diverse',
                'parallel': True,
                'num_solutions': 2
            },
            'evaluation': {
                'num_samples': None,  # Process all samples
                'output_dir': 'results_academic',
                'save_formats': ['json', 'csv']
            }
        },
        'prompt_decorators': {
            'enabled_decorators': ['Reasoning', 'CiteSources', 'FactCheck', 'PeerReview'],
            'decorator_configs': {
                'Reasoning': {'depth': 'comprehensive'},
                'CiteSources': {'style': 'footnote', 'format': 'Chicago', 'comprehensive': True},
                'FactCheck': {'confidence': 0.95, 'strictness': 'high'},
                'PeerReview': {'criteria': ['accuracy', 'clarity', 'completeness', 'originality'], 'style': 'critical'}
            }
        },
        'verification': {
            'agent': {'type': 'robust', 'temperature': 0.1},
            'gaia_verifier': {'type': 'llm', 'temperature': 0.1}
        }
    }
    
    with open('config_academic.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    console.print("[bold green]✓ Academic configuration created: config_academic.yaml[/bold green]")

def demonstrate_config_usage():
    """Demonstrate how to use different configurations."""
    console.print("\n[bold blue]Configuration Usage Examples[/bold blue]")
    console.print("=" * 50)
    
    # Example 1: Fast Test
    console.print("\n[bold yellow]1. Fast Test Configuration[/bold yellow]")
    try:
        config = PlanGENConfig('config_fast_test.yaml')
        config.print_config_summary()
    except FileNotFoundError:
        console.print("[red]Run create_fast_test_config() first[/red]")
    
    # Example 2: Research
    console.print("\n[bold yellow]2. Research Configuration[/bold yellow]")
    try:
        config = PlanGENConfig('config_research.yaml')
        config.print_config_summary()
    except FileNotFoundError:
        console.print("[red]Run create_research_config() first[/red]")
    
    # Example 3: Adaptive Configuration
    console.print("\n[bold yellow]3. Adaptive Configuration Example[/bold yellow]")
    try:
        config = PlanGENConfig('config_research.yaml')
        
        # Test adaptive decorator selection
        math_problem = "Calculate the derivative of f(x) = x^2 + 3x + 1"
        debate_problem = "Debate the pros and cons of artificial intelligence in education"
        
        math_decorators = config.get_adaptive_decorators_for_problem(math_problem)
        debate_decorators = config.get_adaptive_decorators_for_problem(debate_problem)
        
        console.print(f"Math problem decorators: {math_decorators}")
        console.print(f"Debate problem decorators: {debate_decorators}")
        
    except FileNotFoundError:
        console.print("[red]Run create_research_config() first[/red]")

def main():
    """Main function to create example configurations."""
    console.print("[bold blue]PlanGEN Configuration Examples[/bold blue]")
    console.print("=" * 50)
    
    # Create different configuration examples
    create_fast_test_config()
    create_research_config()
    create_creative_config()
    create_academic_config()
    
    # Demonstrate usage
    demonstrate_config_usage()
    
    console.print("\n[bold green]✓ All example configurations created![/bold green]")
    console.print("\n[bold yellow]Usage:[/bold yellow]")
    console.print("1. python plangen_gaia_BestofN_New_prompt_challenger_run1_config.py  # Uses default config")
    console.print("2. Modify the config file to use different settings")
    console.print("3. Or create custom configurations using the examples above")

if __name__ == "__main__":
    main() 