import wave

import openai
import time
import random
from datetime import datetime, timedelta
import requests
import pyttsx3  # Text-to-speech library (alternatively, use another service like Google TTS)
import transformers
import torch
from transformers import pipeline, AutoTokenizer

# Agent personas
# TODO: Automate persona generation (using number of agents)
personas = {
    "Agent1": {"name": "Alice", "persona": "Alice is a friendly and curious researcher."},
    "Agent2": {"name": "Bob", "persona": "Bob is an experienced software engineer who loves discussing coding."},
    "Agent3": {"name": "Charlie", "persona": "Charlie is an empathetic therapist who enjoys deep conversations."},
}

# LLM setup
model_id = "meta-llama/Llama-3.2-1B"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


def prompt_llama(agent, conversation_history, max_new_tokens=512, temperature=0.7):
    """
    Creates a structured prompt for LLaMA-based models with proper response handling
    and length controls.

    Args:
        agent (str): The agent identifier to use for personality
        conversation_history (list): List of dictionaries with role and content keys.
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Controls randomness in generation (0.0 to 1.0)

    Returns:
        str: The generated response without the prompt
    """
    #TODO: Fix conversation history to be a list of dictionaries for the prompting part
    print('prompting llm...')
    agent_name = personas[agent]["name"]
    agent_persona = personas[agent]["persona"]

    prompt_sections = [
        # System instructions section
        f"You are now participating in a conversation as {agent_name}. Provide a complete response based on the following instructions:",
        "- Maintain consistent personality and knowledge based on your character",
        "- Remember context from the conversation history",
        "- Generate responses that align with your character's traits",
        "- Stay in character while being helpful and engaging",
        "- Ensure your response is complete and not cut off",
        "",
        # Character definition section
        f"CHARACTER DEFINITION",
        f"Name: {agent_name}",
        f"Persona: {agent_persona}",
        "",
        # Conversation history (already formatted)
        "CONVERSATION HISTORY:",
        conversation_history,
        "",
        # Add a clear marker for where the response should begin
        f"{agent_name}'s response:"
    ]

    # Join all sections and get response
    final_prompt = "\n".join(prompt_sections)

    # maybe the number of tokens get set dynamically based on how relevant it is to the person?
    new_tokens = random.randrange(30, 71, 8)
    min_tokens = 50
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Update pipeline parameters for longer output
    response = pipe(final_prompt, min_new_tokens=min_tokens, max_new_tokens=new_tokens + min_tokens,
                    stop_strings=[".", "?", "!"], return_full_text=False, num_beams=2, tokenizer=tokenizer)
    # another variable for how long they haven't spoken

    full_text = response[0]['generated_text']


    # Extract only the generated response after the marker
    response_marker = f"{agent_name}'s response:"
    if response_marker in full_text:
        message = full_text.split(response_marker)[-1].strip()
    else:
        message = full_text.strip()

    return message


def sim_speak(text, filename):
    """
    Converts text to speech and saves it to a file while handling the timing accurately.

    Args:
        text (str): The text to convert to speech
        filename (str): Output audio file path
    """
    try:
        engine = pyttsx3.init()

        # Get the speaking rate
        rate = engine.getProperty("rate")

        # Save speech to file
        engine.save_to_file(text, filename)
        engine.runAndWait()

        # Calculate more accurate duration using the audio file
        try:
            with wave.open(filename, 'rb') as audio_file:
                frames = audio_file.getnframes()
                rate = audio_file.getframerate()
                duration = frames / float(rate)
                return duration
        except:
            # Fallback duration calculation if file reading fails
            words = len(text.split())
            return words / (rate / 60)  # Approximate duration in seconds

    finally:
        # Properly dispose of the engine
        engine.stop()

    return 0  # Return 0 if everything fails



# Simulate the conversation
def simulate_conversation(discussion_starter, num_agents=3, duration_minutes=.5):
    conversation_history = discussion_starter
    agents = list(personas.keys())
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    lastSpoke = ""
    agent = "Default"

    turn = 0
    while datetime.now() < end_time:
        # select an agent to speak (shouldn't be the last agent)
        while True:
            agent = random.choice(agents)  # Randomly select an agent to speak
            if lastSpoke != agent:
                agent_name = personas[agent]["name"]
                agent_persona = personas[agent]["persona"]
                lastSpoke = agent
                break

        # gets back text response
        response = prompt_llama(agent, conversation_history)

        # Add agent's response to conversation history
        conversation_history += f"\n{agent_name}: {response}"

        # Convert all responses to audio
        audio_filename = f"testrun/agent_{agent_name}_turn_{turn}.wav"
        sim_speak(response, audio_filename) # will wait appropriate time before cont.

        turn += 1  # Increment the turn count for file naming

if __name__ == "__main__":
    simulate_conversation("what are your thoughts on the current job market?")