"""
LLM Interface for PlanGEN
"""

import os
from typing import Any, Dict, List, Optional, Union

import requests


class LLMInterface:
    """Interface for interacting with language models."""

    def __init__(
        self,
        model_name: str = "llama3:8b",
        temperature: float = 0.7,
        model_type: str = "ollama",
        api_base: Optional[str] = None,
    ):
        """Initialize the LLM interface.

        Args:
            model_name: Name of the model to use
            temperature: Temperature for generation
            model_type: Type of model (e.g., "ollama", "openai")
            api_base: Optional API base URL
        """
        self.model_name = model_name
        self.temperature = temperature
        self.model_type = model_type
        self.api_base = api_base or "http://localhost:11434"  # Default Ollama API endpoint

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using the LLM.

        Args:
            prompt: Prompt to send to the LLM
            system_message: Optional system message
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Generated text
        """
        if self.model_type == "ollama":
            return self._generate_ollama(
                prompt=prompt,
                system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _generate_ollama(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Ollama.

        Args:
            prompt: Prompt to send to Ollama
            system_message: Optional system message
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Generated text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature or self.temperature,
            "stream": False,
        }

        if system_message:
            payload["system"] = system_message

        if max_tokens:
            payload["num_predict"] = max_tokens

        try:
            print(f"\nSending request to Ollama API with payload: {payload}")
            response = requests.post(f"{self.api_base}/api/generate", json=payload)
            
            if response.status_code != 200:
                error_msg = f"Ollama API returned status code {response.status_code}"
                try:
                    error_details = response.json()
                    error_msg += f"\nError details: {error_details}"
                except:
                    error_msg += f"\nResponse text: {response.text}"
                print(f"\n{error_msg}")
                raise RuntimeError(error_msg)
            
            result = response.json()
            print(f"\nReceived response from Ollama API: {result}")
            
            if "error" in result:
                error_msg = f"Ollama API error: {result['error']}"
                print(f"\n{error_msg}")
                raise RuntimeError(error_msg)
            
            if "response" not in result:
                error_msg = f"Unexpected Ollama API response format: {result}"
                print(f"\n{error_msg}")
                raise RuntimeError(error_msg)
            
            return result["response"]
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Ollama API: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_msg += f"\nError details: {error_details}"
                except:
                    error_msg += f"\nResponse text: {e.response.text}"
            print(f"\n{error_msg}")
            raise RuntimeError(error_msg)

    def batch_generate(
        self,
        prompts: List[str],
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            system_message: Optional system message
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            List of generated texts
        """
        return [
            self.generate(prompt, system_message, temperature, max_tokens)
            for prompt in prompts
        ]
