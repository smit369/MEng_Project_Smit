"""
Configuration Loader for PlanGEN GAIA System
Loads and validates configuration from YAML files and provides easy access to settings.
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from rich.console import Console

console = Console()

@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    temperature: float
    model_type: str

@dataclass
class BestOfNConfig:
    """Best of N algorithm configuration."""
    n_plans: int
    sampling_strategy: str
    parallel: bool
    num_solutions: int

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    num_samples: Optional[int]
    output_dir: str
    save_formats: List[str]

@dataclass
class DecoratorConfig:
    """Individual decorator configuration."""
    name: str
    parameters: Dict[str, Any]

@dataclass
class VerificationConfig:
    """Verification configuration."""
    agent_type: str
    agent_model_name: str
    agent_temperature: float
    gaia_verifier_type: str
    gaia_verifier_model_name: str
    gaia_verifier_temperature: float
    verification_decorators: List[str]

@dataclass
class PerformanceConfig:
    """Performance configuration."""
    parallel_enabled: bool
    max_workers: int
    caching_enabled: bool
    cache_dir: str
    max_cache_size: str
    timeouts: Dict[str, int]

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    console_output: bool
    file_output: bool
    log_file: str
    detailed: Dict[str, bool]

class ConfigValidator:
    """Validates configuration parameters against defined rules."""
    
    @staticmethod
    def validate_temperature(value: float) -> bool:
        """Validate temperature parameter."""
        return 0.0 <= value <= 1.0
    
    @staticmethod
    def validate_n_plans(value: int) -> bool:
        """Validate number of plans."""
        return 1 <= value <= 20
    
    @staticmethod
    def validate_socratic_iterations(value: int) -> bool:
        """Validate Socratic iterations."""
        return 1 <= value <= 5
    
    @staticmethod
    def validate_debate_perspectives(value: int) -> bool:
        """Validate debate perspectives."""
        return 1 <= value <= 5
    
    @staticmethod
    def validate_confidence(value: float) -> bool:
        """Validate confidence parameter."""
        return 0.0 <= value <= 1.0
    
    @staticmethod
    def validate_sampling_strategy(value: str) -> bool:
        """Validate sampling strategy."""
        return value in ["basic", "diverse", "adaptive"]
    
    @staticmethod
    def validate_decorator_name(value: str) -> bool:
        """Validate decorator name."""
        valid_decorators = [
            "StepByStep", "Reasoning", "Socratic", "Debate", 
            "PeerReview", "CiteSources", "FactCheck", "Outline",
            "Bullet", "Timeline", "Comparison", "Alternatives"
        ]
        return value in valid_decorators

class PlanGENConfig:
    """Main configuration class for PlanGEN system."""
    
    def __init__(self, config_path: str = "plangen_config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            console.print(f"[bold green]✓ Loaded configuration from {self.config_path}[/bold green]")
            return config
        except FileNotFoundError:
            console.print(f"[bold red]Configuration file {self.config_path} not found![/bold red]")
            raise
        except yaml.YAMLError as e:
            console.print(f"[bold red]Error parsing YAML configuration: {e}[/bold red]")
            raise
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        validator = ConfigValidator()
        
        # Validate core parameters
        plangen_config = self.config.get('plangen', {})
        model_config = plangen_config.get('model', {})
        best_of_n_config = plangen_config.get('best_of_n', {})
        
        # Validate model configuration
        if not validator.validate_temperature(model_config.get('temperature', 0.7)):
            raise ValueError(f"Invalid temperature: {model_config.get('temperature')}")
        
        # Validate Best of N configuration
        if not validator.validate_n_plans(best_of_n_config.get('n_plans', 5)):
            raise ValueError(f"Invalid n_plans: {best_of_n_config.get('n_plans')}")
        
        if not validator.validate_sampling_strategy(best_of_n_config.get('sampling_strategy', 'diverse')):
            raise ValueError(f"Invalid sampling_strategy: {best_of_n_config.get('sampling_strategy')}")
        
        # Validate decorator configurations
        decorator_configs = self.config.get('prompt_decorators', {}).get('decorator_configs', {})
        for decorator_name, config in decorator_configs.items():
            if not validator.validate_decorator_name(decorator_name):
                console.print(f"[bold yellow]Warning: Unknown decorator '{decorator_name}'[/bold yellow]")
            
            # Validate specific decorator parameters
            if decorator_name == 'Socratic' and 'iterations' in config:
                if not validator.validate_socratic_iterations(config['iterations']):
                    raise ValueError(f"Invalid Socratic iterations: {config['iterations']}")
            
            elif decorator_name == 'Debate' and 'perspectives' in config:
                if not validator.validate_debate_perspectives(config['perspectives']):
                    raise ValueError(f"Invalid debate perspectives: {config['perspectives']}")
            
            elif decorator_name == 'FactCheck' and 'confidence' in config:
                if not validator.validate_confidence(config['confidence']):
                    raise ValueError(f"Invalid confidence: {config['confidence']}")
        
        console.print("[bold green]✓ Configuration validation passed[/bold green]")
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        logging_config = self.config.get('logging', {})
        
        # Configure logging level
        log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
        logging.basicConfig(level=log_level)
        
        # Setup file logging if enabled
        if logging_config.get('file_output', True):
            log_file = logging_config.get('log_file', 'plangen.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            logging.getLogger().addHandler(file_handler)
        
        console.print(f"[bold green]✓ Logging configured (level: {logging_config.get('level', 'INFO')})[/bold green]")
    
    # =============================================================================
    # CONFIGURATION ACCESS METHODS
    # =============================================================================
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        model_config = self.config.get('plangen', {}).get('model', {})
        return ModelConfig(
            name=model_config.get('name', 'llama3:8b'),
            temperature=model_config.get('temperature', 0.7),
            model_type=model_config.get('model_type', 'ollama')
        )
    
    def get_best_of_n_config(self) -> BestOfNConfig:
        """Get Best of N configuration."""
        best_of_n_config = self.config.get('plangen', {}).get('best_of_n', {})
        return BestOfNConfig(
            n_plans=best_of_n_config.get('n_plans', 5),
            sampling_strategy=best_of_n_config.get('sampling_strategy', 'diverse'),
            parallel=best_of_n_config.get('parallel', True),
            num_solutions=best_of_n_config.get('num_solutions', 3)
        )
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        eval_config = self.config.get('plangen', {}).get('evaluation', {})
        return EvaluationConfig(
            num_samples=eval_config.get('num_samples'),
            output_dir=eval_config.get('output_dir', 'results'),
            save_formats=eval_config.get('save_formats', ['json', 'csv'])
        )
    
    def get_enabled_decorators(self) -> List[str]:
        """Get list of enabled decorators."""
        # Check for new structure first (challenger_decorators.enabled)
        challenger_decorators = self.config.get('prompt_decorators', {}).get('challenger_decorators', {}).get('enabled', [])
        if challenger_decorators:
            return challenger_decorators
        
        # Fallback to old structure (enabled_decorators)
        return self.config.get('prompt_decorators', {}).get('enabled_decorators', [])
    
    def get_decorator_config(self, decorator_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific decorator."""
        # Check for new structure first (challenger_decorators.configs)
        challenger_configs = self.config.get('prompt_decorators', {}).get('challenger_decorators', {}).get('configs', {})
        if decorator_name in challenger_configs:
            return challenger_configs.get(decorator_name)
        
        # Fallback to old structure (decorator_configs)
        decorator_configs = self.config.get('prompt_decorators', {}).get('decorator_configs', {})
        return decorator_configs.get(decorator_name)
    
    def get_all_decorator_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all decorator configurations."""
        # Check for new structure first (challenger_decorators.configs)
        challenger_configs = self.config.get('prompt_decorators', {}).get('challenger_decorators', {}).get('configs', {})
        if challenger_configs:
            return challenger_configs
        
        # Fallback to old structure (decorator_configs)
        return self.config.get('prompt_decorators', {}).get('decorator_configs', {})
    
    def get_verification_config(self) -> VerificationConfig:
        """Get verification configuration."""
        verification_config = self.config.get('verification', {})
        agent_config = verification_config.get('agent', {})
        gaia_verifier_config = verification_config.get('gaia_verifier', {})
        
        return VerificationConfig(
            agent_type=agent_config.get('type', 'robust'),
            agent_model_name=agent_config.get('model_name', 'llama3:8b'),
            agent_temperature=agent_config.get('temperature', 0.1),
            gaia_verifier_type=gaia_verifier_config.get('type', 'yaml'),
            gaia_verifier_model_name=gaia_verifier_config.get('model_name', 'llama3:8b'),
            gaia_verifier_temperature=gaia_verifier_config.get('temperature', 0.1),
            verification_decorators=self.get_verification_decorators()
        )
    
    def get_early_verification_config(self) -> Dict[str, Any]:
        """Get early verification configuration."""
        verification_config = self.config.get('verification', {})
        early_verification_config = verification_config.get('early_verification', {})
        
        return {
            'enabled': early_verification_config.get('enabled', True),
            'decorators': early_verification_config.get('decorators', ["FactCheck", "CiteSources", "Confidence", "Precision"]),
            'temperature': early_verification_config.get('temperature', 0.1),
            'strictness': early_verification_config.get('strictness', 'medium')
        }
    
    def get_early_verification_decorators(self) -> List[str]:
        """Get list of early verification decorators."""
        early_config = self.get_early_verification_config()
        return early_config.get('decorators', [])
    
    def get_early_verification_decorator_config(self, decorator_name: str) -> Dict[str, Any]:
        """Get configuration for a specific early verification decorator."""
        early_decorators_config = self.config.get('prompt_decorators', {}).get('early_verification_decorators', {})
        configs = early_decorators_config.get('configs', {})
        return configs.get(decorator_name, {})
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        perf_config = self.config.get('performance', {})
        parallel_config = perf_config.get('parallel', {})
        caching_config = perf_config.get('caching', {})
        
        return PerformanceConfig(
            parallel_enabled=parallel_config.get('enabled', True),
            max_workers=parallel_config.get('max_workers', 4),
            caching_enabled=caching_config.get('enabled', True),
            cache_dir=caching_config.get('cache_dir', '.cache'),
            max_cache_size=caching_config.get('max_cache_size', '1GB'),
            timeouts=perf_config.get('timeouts', {})
        )
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        logging_config = self.config.get('logging', {})
        return LoggingConfig(
            level=logging_config.get('level', 'INFO'),
            console_output=logging_config.get('console_output', True),
            file_output=logging_config.get('file_output', True),
            log_file=logging_config.get('log_file', 'plangen.log'),
            detailed=logging_config.get('detailed', {})
        )
    
    def get_experimental_config(self) -> Dict[str, Any]:
        """Get experimental configuration."""
        return self.config.get('experimental', {})
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.config.get('environment', {})
    
    def get_verification_decorators(self) -> list:
        """Get list of enabled verification decorators."""
        return self.config.get('prompt_decorators', {}).get('verification_decorators', {}).get('enabled', [])
    
    def get_decorator_params(self, decorator_name: str, decorator_type: str = 'challenger') -> Dict[str, Any]:
        """Get decorator parameters for a specific decorator and type."""
        if decorator_type == 'challenger':
            # Get from challenger_decorators.configs
            challenger_configs = self.config.get('prompt_decorators', {}).get('challenger_decorators', {}).get('configs', {})
            return challenger_configs.get(decorator_name, {})
        elif decorator_type == 'verification':
            # Get from verification_decorators.configs
            verification_configs = self.config.get('prompt_decorators', {}).get('verification_decorators', {}).get('configs', {})
            return verification_configs.get(decorator_name, {})
        else:
            # Fallback to general decorator configs
            return self.get_decorator_config(decorator_name) or {}
    
    # =============================================================================
    # ADAPTIVE CONFIGURATION METHODS
    # =============================================================================
    
    def get_adaptive_decorators_for_problem(self, problem: str) -> List[str]:
        """Get all enabled challenger decorators (no adaptive selection)."""
        # Simply return all enabled challenger decorators
        return self.get_enabled_decorators()
    
    def get_adaptive_parameters_for_problem(self, problem: str) -> Dict[str, Dict[str, Any]]:
        """Get all decorator configurations (no adaptive parameters)."""
        # Simply return all decorator configurations
        return self.get_all_decorator_configs()
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def print_config_summary(self) -> None:
        """Print a summary of the current configuration."""
        console.print("\n[bold blue]PlanGEN Configuration Summary[/bold blue]")
        console.print("=" * 50)
        
        # Model configuration
        model_config = self.get_model_config()
        console.print(f"Model: {model_config.name} (temp: {model_config.temperature})")
        
        # Best of N configuration
        best_of_n_config = self.get_best_of_n_config()
        console.print(f"Best of N: {best_of_n_config.n_plans} plans, {best_of_n_config.sampling_strategy} strategy")
        
        # Evaluation configuration
        eval_config = self.get_evaluation_config()
        console.print(f"Evaluation: {eval_config.num_samples or 'all'} samples")
        
        # Decorators
        enabled_decorators = self.get_enabled_decorators()
        console.print(f"Enabled Decorators: {', '.join(enabled_decorators)}")
        
        # Performance
        perf_config = self.get_performance_config()
        console.print(f"Performance: Parallel {'Enabled' if perf_config.parallel_enabled else 'Disabled'}")
        
        console.print("=" * 50)
    
    def save_config(self, output_path: str = None) -> None:
        """Save current configuration to a file."""
        if output_path is None:
            output_path = f"plangen_config_backup_{int(time.time())}.yaml"
        
        with open(output_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
        
        console.print(f"[bold green]Configuration saved to {output_path}[/bold green]")

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_config(config_path: str = "plangen_config.yaml") -> PlanGENConfig:
    """Load configuration from file."""
    return PlanGENConfig(config_path)

def create_default_config(output_path: str = "plangen_config_default.yaml") -> None:
    """Create a default configuration file."""
    default_config = {
        'plangen': {
            'model': {
                'name': 'llama3:8b',
                'temperature': 0.7,
                'model_type': 'ollama'
            },
            'best_of_n': {
                'n_plans': 5,
                'sampling_strategy': 'diverse',
                'parallel': True,
                'num_solutions': 3
            },
            'evaluation': {
                'num_samples': 1,
                'output_dir': 'results',
                'save_formats': ['json', 'csv']
            }
        },
        'prompt_decorators': {
            'enabled_decorators': ['StepByStep', 'Reasoning', 'Socratic', 'Debate', 'PeerReview', 'CiteSources', 'FactCheck'],
            'decorator_configs': {
                'StepByStep': {'numbered': True},
                'Reasoning': {'depth': 'comprehensive'},
                'Socratic': {'iterations': 3},
                'Debate': {'perspectives': 3, 'balanced': True},
                'PeerReview': {'criteria': ['accuracy', 'clarity'], 'style': 'constructive'},
                'CiteSources': {'style': 'inline', 'format': 'APA', 'comprehensive': False},
                'FactCheck': {'confidence': 0.8, 'strictness': 'high'}
            }
        }
    }
    
    with open(output_path, 'w') as file:
        yaml.dump(default_config, file, default_flow_style=False, indent=2)
    
    console.print(f"[bold green]Default configuration created at {output_path}[/bold green]")

if __name__ == "__main__":
    # Example usage
    try:
        config = load_config()
        config.print_config_summary()
    except FileNotFoundError:
        console.print("[bold yellow]Configuration file not found. Creating default configuration...[/bold yellow]")
        create_default_config()
        config = load_config("plangen_config_default.yaml")
        config.print_config_summary() 