import openai
import time
import random
from datetime import datetime, timedelta
import requests
import pyttsx3  # Text-to-speech library (alternatively, use another service like Google TTS)
from playsound import playsound # will actually play the sound, and wait before continuing.



# Agent personas
personas = {
    "Agent1": {"name": "Alice", "persona": "Alice is a friendly and curious researcher."},
    "Agent2": {"name": "Bob", "persona": "Bob is an experienced software engineer who loves discussing coding."},
    "Agent3": {"name": "Charlie", "persona": "Charlie is an empathetic therapist who enjoys deep conversations."},
}


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



def sim_speak(text, filename):
    engine = pyttsx3.init()

    # Get the speaking rate (words per minute)
    rate = engine.getProperty("rate")  # Default is typically around 200 words per minute

    # Calculate the duration of the speech
    words = len(text.split())
    speaking_duration = words / (rate / 60)  # Time in seconds

    # Save speech to file
    engine.save_to_file(text, filename)
    engine.runAndWait()  # Wait until the file is generated

    # Wait for the speech duration
    time.sleep(speaking_duration)



# Simulate the conversation
def simulate_conversation(num_agents, duration_minutes=1):
    conversation_history = ""
    agents = list(personas.keys())
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    lastSpoke = ""
    agent = "Default"

    turn = 0
    while datetime.now() < end_time:
        # select an agent to speak (shouldn't be the last agent)
        while agent != lastSpoke:
            agent = random.choice(agents)  # Randomly select an agent to speak
            agent_name = personas[agent]["name"]
            agent_persona = personas[agent]["persona"]

        # TODO: Generate response from the selected agent.
        # Should use the persona for the person.
        response = None


        # Add agent's response to conversation history
        conversation_history += f"\n{agent_name}: {response}"

        # Convert all responses to audio
        audio_filename = f"testrun/agent_{agent_name}_turn_{turn}.mp3"
        sim_speak(response, audio_filename) # will wait appropriate time before cont.

        turn += 1  # Increment the turn count for file naming


if __name__ == "__main__":
    simulate_conversation(num_agents=3, duration_minutes=5)

