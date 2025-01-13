import openai
import random
import time
import requests
import pyttsx3  # Text-to-speech library (alternatively, use another service like Google TTS)

# OpenAI API Key
openai.api_key = "your_openai_api_key"

# Agent personas
personas = {
    "Agent1": {"name": "Alice", "persona": "Alice is a friendly and curious researcher."},
    "Agent2": {"name": "Bob", "persona": "Bob is an experienced software engineer who loves discussing coding."},
    "Agent3": {"name": "Charlie", "persona": "Charlie is an empathetic therapist who enjoys deep conversations."},
}

# Function to generate agent responses using OpenAI API
def generate_agent_response(agent_name, persona, conversation_history):
    prompt = f"The following conversation is between {agent_name} ({persona}). {conversation_history}\n{agent_name}:"
    
    response = openai.Completion.create(
        model="gpt-4",  # Or use a different model if desired
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    
    return response.choices[0].text.strip()





def prompt_llama(api_url, api_key, prompt_text, max_tokens=100, temperature=0.7):
    """
    Sends a prompt to the Llama model and retrieves the response.

    Parameters:
        api_url (str): The API endpoint for the Llama model.
        api_key (str): Your API key for authentication.
        prompt_text (str): The text prompt to send to the model.
        max_tokens (int): The maximum number of tokens to generate. Default is 100.
        temperature (float): Sampling temperature for response creativity. Default is 0.7.

    Returns:
        str: The generated response from the model.


    # # Example usage:
# api_url = "https://api.llama-model.com/v1/completions"  # Replace with actual API URL
# api_key = "your-api-key"  # Replace with your API key
# prompt_text = "Once upon a time in a magical forest, there lived a..."
#
# response = prompt_llama(api_url, api_key, prompt_text)
# print("Llama's Response:", response)


    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("text", "")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return ""


# Function to convert text to speech (audio)
def text_to_speech(text, filename):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()

# Simulate the conversation
def simulate_conversation(num_agents, max_turns=20):
    conversation_history = ""
    agents = list(personas.keys())
    
    for turn in range(max_turns):
        agent = random.choice(agents)  # Randomly select an agent to speak
        agent_name = personas[agent]["name"]
        agent_persona = personas[agent]["persona"]

        # Generate response from the selected agent
        response = generate_agent_response(agent_name, agent_persona, conversation_history)
        
        # Add agent's response to conversation history
        conversation_history += f"\n{agent_name}: {response}"

        # Convert the response to audio
        audio_filename = f"agent_{agent_name}_turn_{turn}.mp3"
        text_to_speech(response, audio_filename)

        # Simulate interruptions with random chance
        if random.random() < 0.2:  # 20% chance for an interruption
            interrupting_agent = random.choice(agents)
            while interrupting_agent == agent:  # Ensure the interrupter is not the same agent
                interrupting_agent = random.choice(agents)
            
            interrupt_response = generate_agent_response(
                personas[interrupting_agent]["name"],
                personas[interrupting_agent]["persona"],
                conversation_history
            )
            
            # Update history and audio for the interrupting agent
            conversation_history += f"\n{personas[interrupting_agent]['name']}: {interrupt_response}"
            interrupt_audio_filename = f"agent_{personas[interrupting_agent]['name']}_interrupt_{turn}.mp3"
            text_to_speech(interrupt_response, interrupt_audio_filename)

        time.sleep(1)  # Wait for a moment before the next turn

if __name__ == "__main__":
    simulate_conversation(num_agents=3, max_turns=10)
