import json
from typing import Any, Dict, List
from pathlib import Path

def save_json(data: Any, output_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        output_path: Path to save JSON file
        indent: Number of spaces for indentation
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(input_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Any: Loaded JSON data
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_schema(data: Dict, schema: Dict) -> List[str]:
    """
    Validate data against a JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema
        
    Returns:
        List[str]: List of validation errors, empty if valid
    """
    errors = []
    
    def _validate_type(value: Any, expected_type: str, path: str) -> None:
        """Helper to validate type."""
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        if expected_type in type_map:
            if not isinstance(value, type_map[expected_type]):
                errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
    
    def _validate_object(obj: Dict, schema_props: Dict, path: str) -> None:
        """Validate an object against schema properties."""
        # Check required properties
        required = schema.get('required', [])
        for prop in required:
            if prop not in obj:
                errors.append(f"{path}: missing required property '{prop}'")
        
        # Validate each property
        for key, value in obj.items():
            if key in schema_props:
                prop_schema = schema_props[key]
                prop_path = f"{path}.{key}"
                
                _validate_type(value, prop_schema.get('type', 'any'), prop_path)
                
                if prop_schema.get('type') == 'array':
                    if 'items' in prop_schema:
                        for i, item in enumerate(value):
                            _validate_type(item, prop_schema['items'].get('type', 'any'),
                                        f"{prop_path}[{i}]")
    
    # Start validation
    if 'type' in schema:
        _validate_type(data, schema['type'], '$')
        
        if schema['type'] == 'object' and 'properties' in schema:
            _validate_object(data, schema['properties'], '$')
    
    return errors
