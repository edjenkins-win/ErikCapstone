"""Natural language processing module for the Photo Editor application.

This module handles natural language instructions for photo editing by parsing
descriptions into specific editing operations using function calling.
"""

import logging
import json
import os
from typing import Dict, List, Any, Tuple, Optional, Callable
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class NLProcessor:
    """Processes natural language instructions for photo editing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the natural language processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._validate_config()
        
        # Function registry maps function names to actual functions
        self.function_registry = {}
        
    def _validate_config(self) -> None:
        """Validate and set default configuration parameters."""
        defaults = {
            'api_key': os.environ.get('OPENAI_API_KEY', ''),
            'model': 'gpt-4',
            'max_tokens': 150,
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def register_function(self, name: str, func: Callable, description: str, parameters: Dict[str, Any]):
        """Register a function that can be called from natural language.
        
        Args:
            name: Function name
            func: Function to call
            description: Description of what the function does
            parameters: Parameter schema for the function
        """
        self.function_registry[name] = {
            'function': func,
            'description': description,
            'parameters': parameters
        }
    
    def process(self, image: np.ndarray, instruction: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a natural language instruction and apply it to an image.
        
        Args:
            image: Input image
            instruction: Natural language instruction
            
        Returns:
            Tuple of (processed image, metadata)
        """
        # Extract operations from instruction
        operations = self._parse_instruction(instruction)
        
        # Initialize metadata
        metadata = {
            'instruction': instruction,
            'functions_called': [],
            'errors': []
        }
        
        # Apply each operation in sequence
        result = image.copy()
        for op in operations:
            try:
                # Log the function call
                metadata['functions_called'].append({
                    'name': op['name'],
                    'args': op['arguments']
                })
                
                # Call the function
                if op['name'] in self.function_registry:
                    func_info = self.function_registry[op['name']]
                    func = func_info['function']
                    result = func(result, **op['arguments'])
                else:
                    metadata['errors'].append(f"Unknown function: {op['name']}")
                    
            except Exception as e:
                metadata['errors'].append(f"Error in {op['name']}: {str(e)}")
                logger.error(f"Error applying operation {op['name']}: {e}")
        
        return result, metadata
    
    def _parse_instruction(self, instruction: str) -> List[Dict[str, Any]]:
        """Parse natural language instruction into specific operations.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            List of operations (function name and arguments)
        """
        try:
            # In a real implementation, this would call an LLM API with function calling
            # For demo purposes, we'll simulate the function calling behavior
            
            # Generate JSON schema for all registered functions
            tools = []
            for name, info in self.function_registry.items():
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info['description'],
                        "parameters": info['parameters']
                    }
                })
            
            # For demo purposes, let's simulate an LLM response based on the instruction
            return self._simulate_function_calls(instruction)
            
        except Exception as e:
            logger.error(f"Error parsing instruction: {e}")
            return []
    
    def _simulate_function_calls(self, instruction: str) -> List[Dict[str, Any]]:
        """Simulate function calls based on the instruction (demo only).
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            List of simulated function calls
        """
        instruction_lower = instruction.lower()
        operations = []
        
        # Handle brightness/exposure adjustments
        if any(term in instruction_lower for term in ['bright', 'exposure', 'darker', 'lighter']):
            amount = 0.3 if 'bright' in instruction_lower or 'lighter' in instruction_lower else -0.3
            
            # Adjust magnitude based on modifiers
            if 'slightly' in instruction_lower or 'subtle' in instruction_lower:
                amount *= 0.5
            elif 'very' in instruction_lower or 'much' in instruction_lower:
                amount *= 1.5
                
            operations.append({
                'name': 'adjust_exposure',
                'arguments': {'amount': amount}
            })
        
        # Handle contrast adjustments
        if 'contrast' in instruction_lower:
            increase = 'increase' in instruction_lower or 'more' in instruction_lower
            amount = 1.2 if increase else 0.8
            
            # Adjust magnitude based on modifiers
            if 'slightly' in instruction_lower or 'subtle' in instruction_lower:
                amount = 1.1 if increase else 0.9
            elif 'very' in instruction_lower or 'much' in instruction_lower or 'dramatic' in instruction_lower:
                amount = 1.4 if increase else 0.7
                
            operations.append({
                'name': 'adjust_contrast',
                'arguments': {'multiplier': amount}
            })
        
        # Handle saturation/vibrance adjustments
        if any(term in instruction_lower for term in ['saturation', 'vibrance', 'vibrant', 'colorful']):
            increase = not ('reduce' in instruction_lower or 'less' in instruction_lower)
            amount = 0.2 if increase else -0.2
            
            # Adjust magnitude based on modifiers
            if 'slightly' in instruction_lower or 'subtle' in instruction_lower:
                amount *= 0.5
            elif 'very' in instruction_lower or 'much' in instruction_lower:
                amount *= 1.5
                
            operations.append({
                'name': 'adjust_saturation',
                'arguments': {'adjustment': amount}
            })
        
        # Handle temperature/warmth adjustments
        if any(term in instruction_lower for term in ['warm', 'temperature', 'cool', 'cold']):
            warm = 'warm' in instruction_lower
            amount = 0.15 if warm else -0.15
            
            # Adjust magnitude based on modifiers
            if 'slightly' in instruction_lower or 'subtle' in instruction_lower:
                amount *= 0.5
            elif 'very' in instruction_lower or 'much' in instruction_lower:
                amount *= 1.5
                
            operations.append({
                'name': 'adjust_temperature',
                'arguments': {'adjustment': amount}
            })
        
        # Handle sharpness adjustments
        if any(term in instruction_lower for term in ['sharp', 'clarity', 'detail']):
            increase = not ('reduce' in instruction_lower or 'less' in instruction_lower)
            amount = 0.3 if increase else -0.1
            
            # Adjust magnitude based on modifiers
            if 'slightly' in instruction_lower or 'subtle' in instruction_lower:
                amount *= 0.7
            elif 'very' in instruction_lower or 'much' in instruction_lower:
                amount *= 1.5
                
            operations.append({
                'name': 'adjust_sharpness',
                'arguments': {'strength': max(0, amount)}
            })
            
        # Handle noise reduction
        if 'noise' in instruction_lower or 'grain' in instruction_lower:
            reduce = 'reduce' in instruction_lower or 'less' in instruction_lower or 'remove' in instruction_lower
            amount = 0.4 if reduce else 0.1
            
            # Adjust magnitude based on modifiers
            if 'slightly' in instruction_lower or 'subtle' in instruction_lower:
                amount *= 0.7
            elif 'very' in instruction_lower or 'much' in instruction_lower:
                amount *= 1.5
                
            operations.append({
                'name': 'reduce_noise',
                'arguments': {'strength': amount}
            })
            
        # Handle style-based instructions
        if 'cinematic' in instruction_lower:
            if 'dramatic' in instruction_lower or 'dark' in instruction_lower:
                operations.append({
                    'name': 'apply_style',
                    'arguments': {'style_name': 'Film Noir'}
                })
            elif 'anamorphic' in instruction_lower or 'widescreen' in instruction_lower:
                operations.append({
                    'name': 'apply_style',
                    'arguments': {'style_name': 'Anamorphic'}
                })
            else:
                operations.append({
                    'name': 'apply_style',
                    'arguments': {'style_name': 'Cinematic Teal & Orange'}
                })
        elif 'vintage' in instruction_lower or 'retro' in instruction_lower:
            operations.append({
                'name': 'apply_style',
                'arguments': {'style_name': 'Vintage'}
            })
        elif 'portrait' in instruction_lower:
            operations.append({
                'name': 'apply_style',
                'arguments': {'style_name': 'Portrait'}
            })
        elif 'dreamy' in instruction_lower or 'soft' in instruction_lower:
            operations.append({
                'name': 'apply_style',
                'arguments': {'style_name': 'Dreamy'}
            })
        elif 'dramatic' in instruction_lower or 'action' in instruction_lower:
            operations.append({
                'name': 'apply_style',
                'arguments': {'style_name': 'Blockbuster'}
            })
            
        # If no specific operations were identified, apply auto-enhance
        if not operations:
            operations.append({
                'name': 'apply_style',
                'arguments': {'style_name': 'Auto-Enhance'}
            })
            
        return operations 