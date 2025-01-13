import openai
import time
import random
from datetime import datetime, timedelta
import requests
import pyttsx3  # Text-to-speech library (alternatively, use another service like Google TTS)
import transformers
import torch



# Agent personas
# TODO: Automate persona generation
personas = {
    "Agent1": {"name": "Alice", "persona": "Alice is a friendly and curious researcher."},
    "Agent2": {"name": "Bob", "persona": "Bob is an experienced software engineer who loves discussing coding."},
    "Agent3": {"name": "Charlie", "persona": "Charlie is an empathetic therapist who enjoys deep conversations."},
}

# LLM setup
model_id = "meta-llama/Meta-Llama-3-8B"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipeline("Hey how are you doing today?")

def prompt_llama(message, persona):
    "responsible for prompting LLM"

    # TODO: Add more robust prompting

    # add persona to message
    if persona:
        message = persona + message

    return pipeline(message)



def sim_speak(text, filename):
    engine = pyttsx3.init()

    # Get the speaking rate (words per minute)
    rate = engine.getProperty("rate")  # Default is typically around 200 words per minute

    # also needs to consider things like tone, emotion, speaking rate. Ideally this would happen later later?

    # Calculate the duration of the speech
    words = len(text.split())
    speaking_duration = words / (rate / 60)  # Time in seconds

    # Save speech to file
    engine.save_to_file(text, filename)
    engine.runAndWait()  # Wait until the file is generated

    # Wait for the speech duration
    time.sleep(speaking_duration)



# Simulate the conversation
def simulate_conversation(num_agents=3, duration_minutes=1):
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
    simulate_conversation()

