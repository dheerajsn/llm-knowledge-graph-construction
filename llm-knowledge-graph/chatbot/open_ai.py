import os
import json
import datetime
from openai import OpenAI
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define tool/function schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in a specific timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name, e.g., 'America/New_York'"
                    }
                },
                "required": []
            }
        }
    }
]

# Implement the actual tool functions
def get_weather(location, unit="celsius"):
    """Get weather for a location (simulated)"""
    # In a real application, you would call a weather API
    weather_data = {
        "San Francisco": {"temperature": 18, "conditions": "Foggy"},
        "New York": {"temperature": 22, "conditions": "Sunny"},
        "London": {"temperature": 15, "conditions": "Rainy"},
    }
    
    # Extract the city name from location
    city = location.split(",")[0].strip()
    
    # Get weather data or default
    data = weather_data.get(city, {"temperature": 20, "conditions": "Clear"})
    
    # Convert temperature if needed
    temp = data["temperature"]
    if unit == "fahrenheit":
        temp = (temp * 9/5) + 32
    
    return {
        "location": location,
        "temperature": temp,
        "temperature_unit": unit,
        "conditions": data["conditions"]
    }

def get_current_time(timezone=None):
    """Get current time in specified timezone (or local time)"""
    now = datetime.datetime.now()
    return {
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone or "local"
    }

# Function dispatcher
def execute_function(name, arguments):
    function_map = {
        "get_weather": get_weather,
        "get_current_time": get_current_time
    }
    
    if name not in function_map:
        return {"error": f"Function {name} not found"}
    
    try:
        func = function_map[name]
        return func(**arguments)
    except Exception as e:
        return {"error": f"Error executing {name}: {str(e)}"}

# Process user query with tool calling
def process_query(user_input):
    messages = [{"role": "user", "content": user_input}]
    
    # Step 1: Call the model with the user query and available tools
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",  # or another model with function calling support
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Let the model decide when to call functions
    )
    
    # Get the response message
    response_message = response.choices[0].message
    
    # Add assistant's message to the conversation
    messages.append(response_message)
    
    # Step 2: Check if the model wanted to call a function
    if response_message.tool_calls:
        # Process each tool call
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Print the function call for debugging
            print(f"Function call: {function_name}({function_args})")
            
            # Execute the function
            function_response = execute_function(function_name, function_args)
            
            # Append the function response to the messages
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response)
            })
        
        # Step 3: Call the model again with the function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
        )
        
        # Return the final response
        return second_response.choices[0].message.content
    
    # If no function was called, return the original response
    return response_message.content

# Example usage
if __name__ == "__main__":
    # Test with different queries
    test_queries = [
        "What's the weather like in San Francisco?",
        "What time is it now?",
        "Should I bring an umbrella if I'm going to London tomorrow?",
        "What's the temperature in New York in Fahrenheit?"
    ]
    
    for query in test_queries:
        print(f"\n\nQuery: {query}")
        print("=" * 50)
        result = process_query(query)
        print(f"Response: {result}")
        print("=" * 50)