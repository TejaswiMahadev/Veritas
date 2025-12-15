import os
import requests
import json
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

load_dotenv()

class NimClient:
    """
    Wrapper for Nvidia NIM API interactions.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is not set in environment variables.")
        
        self.base_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.model = "google/gemma-3n-e4b-it"
        
    def generate(
        self, 
        prompt: str, 
        system_instruction: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = True
    ) -> Union[Dict[str, Any], str]:
        """
        Generate a response from the NIM model.
        
        Args:
            prompt: The user prompt.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            json_mode: If True, tries to ensure JSON output (via prompt instruction).
        
        Returns:
            Parsed JSON dict if json_mode is True and successful, else raw string response.
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.70,
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            if json_mode:
                return self._clean_and_parse_json(content)
            
            return content
            
        except requests.exceptions.RequestException as e:
            print(f"NIM API Request Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            raise
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing NIM response: {e}")
            # Return raw content if parsing fails but request succeeded
            if 'content' in locals():
                return content
            raise

    def _clean_and_parse_json(self, content: str) -> Dict[str, Any]:
        """Helper to extract JSON from markdown code blocks or raw text."""
        import re
        try:
            # Try direct parse first
            return json.loads(content, strict=False)
        except json.JSONDecodeError:
            pass

        try:
            # Remove markdown code blocks
            cleaned = content.strip()
            if "```" in cleaned:
                # patterns like ```json ... ``` or just ``` ... ```
                pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
                match = re.search(pattern, cleaned, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
            
            # Fallback: Find the first outer set of braces
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
                
            raise valueError(f"Could not find JSON in response: {content[:100]}...")
            
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            raise json.JSONDecodeError("Failed to extract JSON", content, 0)
