#!/usr/bin/env python3
"""
Test script to verify the new prompt decorators framework is working correctly.
"""

import prompt_decorators as pd
from prompt_decorators import (
    load_decorator_definitions,
    get_available_decorators,
    create_decorator_instance,
    apply_decorator,
    apply_dynamic_decorators
)

def test_prompt_decorators():
    """Test the prompt decorators framework."""
    print("Testing Prompt Decorators Framework...")
    
    # Load decorator definitions
    load_decorator_definitions()
    print("✓ Loaded decorator definitions")
    
    # Get available decorators
    decorators = get_available_decorators()
    print(f"✓ Found {len(decorators)} available decorators")
    
    # Test specific decorators we plan to use
    test_decorators = ['StepByStep', 'Reasoning', 'Socratic', 'Debate', 'PeerReview', 'CiteSources', 'FactCheck']
    
    test_prompt = "Explain how photosynthesis works."
    
    print("\nTesting individual decorators:")
    for decorator_name in test_decorators:
        try:
            # Test if decorator exists and can be applied
            enhanced_prompt = apply_decorator(decorator_name, test_prompt)
            print(f"✓ {decorator_name}: Success")
            print(f"  Original: {test_prompt}")
            print(f"  Enhanced: {enhanced_prompt[:100]}...")
            print()
        except Exception as e:
            print(f"✗ {decorator_name}: Error - {str(e)}")
            print()
    
    # Test dynamic decorators with +++ syntax
    print("Testing dynamic decorators with +++ syntax:")
    dynamic_prompt = "+++StepByStep\n+++Reasoning\nExplain quantum mechanics."
    try:
        enhanced = apply_dynamic_decorators(dynamic_prompt)
        print(f"✓ Dynamic decorators: Success")
        print(f"  Original: {dynamic_prompt}")
        print(f"  Enhanced: {enhanced[:100]}...")
    except Exception as e:
        print(f"✗ Dynamic decorators: Error - {str(e)}")
    
    print("\n✓ Prompt decorators framework test completed!")

if __name__ == "__main__":
    test_prompt_decorators() 