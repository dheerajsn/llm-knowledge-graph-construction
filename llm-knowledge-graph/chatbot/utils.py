import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

# tag::write_message[]
def write_message(role, content, save = True):
    """
    This is a helper function that saves a message to the
     session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write to UI
    with st.chat_message(role):
        st.markdown(content)
# end::write_message[]

# tag::get_session_id[]
def get_session_id():
    return get_script_run_ctx().session_id
# end::get_session_id[]

import re
from datetime import datetime

def parse_string_to_dict(text):
    """Parse key-value string to dictionary using regex"""
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Pattern to match key: value pairs
    pattern = r'(\w+):\s*([^,]+?)(?=\s+\w+:|$)'
    matches = re.findall(pattern, text)
    
    result = {}
    for key, value in matches:
        value = value.strip().strip('"\'')
        
        # Convert values
        if value.lower() == 'null':
            result[key] = None
        elif value.isdigit():
            result[key] = int(value)
        elif value.replace('.', '').isdigit():
            result[key] = float(value)
        else:
            result[key] = value
    
    return result

# Test strings
string1 = '''start_catetime: "2025-04-07"
end _time: "2025-04-13"
desk_name: "AMRS LINEAR RATES", trader_names: null, alert_type: null, data_source: null, limit: 100'''

string2 = '''start_datetime: 2025-04-30T23:59:59Z
end_datetime: 2025-04-30T23:59:59Z
desk_name: AMRS LINEAR RATES
trader_names: null 
alert_type: null 
data_source: null limit: 10'''

# Parse both strings
dict1 = parse_string_to_dict(string1)
dict2 = parse_string_to_dict(string2)

print("Dictionary 1:")
print(dict1)
print("\nDictionary 2:")
print(dict2)