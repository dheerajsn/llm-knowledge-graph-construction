import re
from typing import Dict, Any

def parse_config_string(text: str) -> Dict[str, Any]:
    """Parse configuration-like strings to dictionary"""
    
    def convert_value(value: str) -> Any:
        """Convert string value to appropriate type"""
        # Check if the value is null
        if value.lower() == 'null':
            return None
        
        # Check if it's a number
        try:
            # Try integer first
            if value.isdigit():
                return int(value)
            # Then try float
            return float(value)
        except ValueError:
            pass
            
        # Otherwise return as string
        return value
    
    # Remove curly braces if present
    text = text.strip()
    if text.startswith('{'):
        text = text[1:]
    if text.endswith('}'):
        text = text[:-1]
    
    result = {}
    
    # Process each line separately
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        # First handle comma-separated parts
        if ',' in line:
            parts = line.split(',')
        else:
            parts = [line]
        
        for part in parts:
            part = part.strip()
            if not part or ':' not in part:
                continue
            
            # Extract key and value
            key, value = part.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Handle quoted values - remove quotes
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            # Add to result dict
            if value:
                result[key] = convert_value(value)
    
    # Special handling for space-separated formats in string2
    # Look for patterns like "data_source: null limit: 10" where multiple key:value pairs are on one line
    for key, value in list(result.items()):
        if isinstance(value, str):
            # Check if the value contains patterns like "key: value"
            key_value_pattern = re.search(r'\s+(\w+)\s*:\s*(\S+)', value)
            if key_value_pattern:
                # Extract the real value part and the embedded key-value pair
                embedded_key = key_value_pattern.group(1)
                embedded_value = key_value_pattern.group(2)
                
                # Update the original value to exclude the embedded key-value
                real_value = value[:key_value_pattern.start()].strip()
                result[key] = convert_value(real_value) if real_value else None
                
                # Add the embedded key-value
                result[embedded_key] = convert_value(embedded_value)
    
    return result

# Test strings
string1 = '''
start_catetime: "2025-04-07",
end_time: "2025-04-13",
desk_name: "AMRS LINEAR RATES",
trader_names: null,
alert_type: null,
data_source: null,
limit: 100
'''

string2 = '''start_datetime: 2025-04-30T23:59:59Z
end_datetime: 2025-04-30T23:59:59Z
desk_name: AMRS LINEAR RATES
trader_names: null 
alert_type: null 
data_source: null limit: 10'''

# Parse both strings
dict1 = parse_config_string(string1)
dict2 = parse_config_string(string2)

print("Dictionary 1:")
print(dict1)
print("\nDictionary 2:")
print(dict2)