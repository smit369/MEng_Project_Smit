# PlanGEN GAIA Configuration Guide

This guide explains how to use the comprehensive configuration system for the PlanGEN GAIA evaluation system with prompt decorators.

## 📁 Files Overview

- **`plangen_config.yaml`** - Main configuration file (YAML format)
- **`config_loader.py`** - Python configuration loader and validator
- **`plangen_gaia_BestofN_New_prompt_challenger_run1_config.py`** - Updated main script that uses configuration
- **`example_configurations.py`** - Example configurations for different use cases

## 🚀 Quick Start

### 1. Basic Usage
```bash
# Run with default configuration
python plangen_gaia_BestofN_New_prompt_challenger_run1_config.py
```

### 2. Test Configuration System
```bash
# Test the configuration loader
python config_loader.py

# Create example configurations
python example_configurations.py
```

## 📋 Configuration Structure

The configuration file is organized into several sections:

### Core PlanGEN Configuration
```yaml
plangen:
  model:
    name: "llama3:8b"           # LLM model to use
    temperature: 0.7            # Creativity level (0.0-1.0)
    model_type: "ollama"        # Model interface type
  
  best_of_n:
    n_plans: 5                  # Number of initial plans
    sampling_strategy: "diverse" # "basic", "diverse", "adaptive"
    parallel: true              # Enable parallel processing
    num_solutions: 3            # Number of final solutions
  
  evaluation:
    num_samples: 1              # Number of GAIA samples (null = all)
    output_dir: "results"       # Output directory
    save_formats: ["json", "csv"] # Output formats
```

### Prompt Decorator Configuration
```yaml
prompt_decorators:
  enabled_decorators:
    - "StepByStep"
    - "Reasoning"
    - "Socratic"
    - "Debate"
    - "PeerReview"
    - "CiteSources"
    - "FactCheck"
  
  decorator_configs:
    StepByStep:
      numbered: true
    
    Reasoning:
      depth: "comprehensive"  # "basic", "moderate", "comprehensive"
    
    Socratic:
      iterations: 3          # Number of question-answer cycles (1-5)
    
    Debate:
      perspectives: 3        # Number of perspectives (1-5)
      balanced: true         # Ensure balanced viewpoints
    
    PeerReview:
      criteria: ["accuracy", "clarity", "completeness"]
      style: "constructive"  # "constructive", "critical"
      position: "neutral"    # "supportive", "neutral", "critical"
    
    CiteSources:
      style: "inline"        # "inline", "footnote", "endnote"
      format: "APA"          # "APA", "MLA", "Chicago", "Harvard", "IEEE"
      comprehensive: false   # Cite every claim (true/false)
    
    FactCheck:
      confidence: 0.8        # Confidence threshold (0.0-1.0)
      uncertain: true        # Flag uncertain claims
      strictness: "high"     # "low", "medium", "high"
```

## 🎯 Pre-configured Examples

The system comes with several pre-configured examples:

### 1. Fast Test Configuration (`config_fast_test.yaml`)
- **Purpose**: Quick testing and development
- **Features**: 
  - 3 plans, basic sampling strategy
  - Only essential decorators (StepByStep, Reasoning)
  - Lower temperature (0.3) for focused responses
  - 1 sample evaluation

### 2. Research Configuration (`config_research.yaml`)
- **Purpose**: Comprehensive research and analysis
- **Features**:
  - 10 plans, diverse sampling strategy
  - All decorators enabled with comprehensive settings
  - Adaptive configuration enabled
  - Full dataset evaluation
  - Detailed logging

### 3. Creative Configuration (`config_creative.yaml`)
- **Purpose**: Creative problem solving
- **Features**:
  - High temperature (0.9) for creativity
  - 8 plans for diversity
  - Less structured decorators
  - Unbalanced debate perspectives

### 4. Academic Configuration (`config_academic.yaml`)
- **Purpose**: Academic rigor and citations
- **Features**:
  - Comprehensive reasoning and fact-checking
  - Chicago-style citations with footnotes
  - Critical peer review
  - High confidence thresholds

## 🔧 Using Different Configurations

### Method 1: Modify the main config file
```bash
# Edit the main configuration
nano plangen_config.yaml

# Run with modified settings
python plangen_gaia_BestofN_New_prompt_challenger_run1_config.py
```

### Method 2: Use specific configuration files
```python
# In your Python code
from config_loader import load_config
from plangen_gaia_BestofN_New_prompt_challenger_run1_config import PlanGENGaia

# Load specific configuration
config = load_config("config_research.yaml")
plangen = PlanGENGaia(config=config)
results = plangen.evaluate_dataset()
```

### Method 3: Create custom configurations
```python
# Create your own configuration
from example_configurations import create_fast_test_config
create_fast_test_config()

# Use the created configuration
config = load_config("config_fast_test.yaml")
```

## 🧠 Adaptive Configuration

The system supports adaptive configuration that adjusts decorators and parameters based on problem characteristics:

### Problem Type Detection
```yaml
adaptive_config:
  enabled: true
  problem_types:
    math:
      decorators: ["StepByStep", "Reasoning", "FactCheck"]
      reasoning_depth: "comprehensive"
      fact_check_strictness: "high"
    
    debate:
      decorators: ["Debate", "PeerReview", "CiteSources"]
      debate_perspectives: 4
      citation_comprehensive: true
    
    analysis:
      decorators: ["Socratic", "Reasoning", "FactCheck"]
      socratic_iterations: 5
      reasoning_depth: "comprehensive"
```

### Length-Based Adjustments
```yaml
length_based:
  short:  # < 200 characters
    reasoning_depth: "basic"
    socratic_iterations: 2
    citation_comprehensive: false
  
  medium:  # 200-500 characters
    reasoning_depth: "moderate"
    socratic_iterations: 3
    citation_comprehensive: false
  
  long:  # > 500 characters
    reasoning_depth: "comprehensive"
    socratic_iterations: 5
    citation_comprehensive: true
```

## 🔍 Value Map Configurations

The system uses value maps to convert parameter values into specific instructions:

### Example: Reasoning Decorator
```yaml
Reasoning:
  depth: "comprehensive"  # This maps to:
  # "Provide a very thorough and detailed analysis with multiple perspectives."
```

### Example: CiteSources Decorator
```yaml
CiteSources:
  style: "footnote"       # Maps to: "Use numbered footnotes for citations..."
  format: "Chicago"       # Maps to: "Format citations according to Chicago..."
  comprehensive: true     # Maps to: "Cite every factual claim..."
```

## ⚡ Performance Configuration

```yaml
performance:
  parallel:
    enabled: true
    max_workers: 4
  
  caching:
    enabled: true
    cache_dir: ".cache"
    max_cache_size: "1GB"
  
  timeouts:
    plan_generation: 300     # Seconds per plan
    verification: 120        # Seconds per verification
    total_per_sample: 1800   # Total seconds per sample
```

## 📊 Logging Configuration

```yaml
logging:
  level: "INFO"              # "DEBUG", "INFO", "WARNING", "ERROR"
  console_output: true
  file_output: true
  log_file: "plangen.log"
  
  detailed:
    log_decorator_applications: true
    log_verification_scores: true
    log_plan_generation: true
    log_timing: true
```

## 🔐 Environment Configuration

```yaml
environment:
  api_keys:
    openai: null              # Set your OpenAI API key here
    anthropic: null           # Set your Anthropic API key here
    huggingface: null         # Set your HuggingFace token here
  
  endpoints:
    ollama: "http://localhost:11434"
    openai: "https://api.openai.com/v1"
    anthropic: "https://api.anthropic.com"
  
  dataset:
    name: "gaia-benchmark/GAIA"
    split: "validation"
    subset: "2023_all"
```

## ✅ Validation Rules

The system automatically validates configuration parameters:

```yaml
validation:
  rules:
    temperature:
      min: 0.0
      max: 1.0
      default: 0.7
    
    n_plans:
      min: 1
      max: 20
      default: 5
    
    socratic_iterations:
      min: 1
      max: 5
      default: 3
```

## 🛠️ Advanced Usage

### Custom Value Maps
```yaml
experimental:
  custom_value_maps:
    Reasoning:
      depth:
        minimal: "Provide only essential reasoning steps."
        standard: "Include clear reasoning with key insights."
        detailed: "Provide comprehensive reasoning with examples."
        expert: "Include expert-level analysis with deep insights."
```

### Decorator Combinations
```yaml
experimental:
  decorator_combinations:
    - name: "Academic"
      decorators: ["Reasoning", "CiteSources", "FactCheck"]
      params:
        reasoning_depth: "comprehensive"
        citation_comprehensive: true
        fact_check_strictness: "high"
```

## 🚨 Troubleshooting

### Common Issues

1. **Configuration file not found**
   ```bash
   # Create default configuration
   python config_loader.py
   ```

2. **Invalid parameter values**
   - Check the validation rules in the configuration
   - Ensure parameters are within the specified ranges

3. **Decorator not found**
   - Verify decorator names match the available decorators
   - Check the prompt-decorators library installation

### Debug Mode
```yaml
logging:
  level: "DEBUG"
  detailed:
    log_decorator_applications: true
    log_verification_scores: true
```

## 📈 Performance Tips

### For Fast Testing
- Use `n_plans: 3` and `sampling_strategy: "basic"`
- Enable only essential decorators
- Set `temperature: 0.3` for focused responses
- Use `parallel: true` with `max_workers: 2`

### For Research
- Use `n_plans: 10` and `sampling_strategy: "diverse"`
- Enable all decorators with comprehensive settings
- Set `temperature: 0.7` for balanced creativity
- Use `parallel: true` with `max_workers: 8`

### For Production
- Use adaptive configuration
- Set appropriate timeouts
- Enable caching
- Use detailed logging for monitoring

## 🔄 Migration from Old Code

If you're migrating from the old hardcoded configuration:

1. **Extract current settings** from your old code
2. **Create configuration file** using the examples
3. **Update imports** to use the new configuration system
4. **Test with small samples** first
5. **Gradually increase complexity**

## 📚 Additional Resources

- **Prompt Decorators Documentation**: [prompt-decorators.ai](https://promptdecorators.ai)
- **PlanGEN Documentation**: Check the main PlanGEN repository
- **YAML Syntax**: [yaml.org](https://yaml.org)

---

This configuration system provides a powerful, flexible interface for customizing every aspect of the PlanGEN GAIA evaluation system while maintaining the simplicity of the original implementation. 