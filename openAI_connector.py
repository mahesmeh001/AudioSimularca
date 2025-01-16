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
    "Agent1": {"name": "Joe",
               "persona": "Joe is a 68-year-old retired factory worker living in a small town. While he’s no longer "
                          "in the workforce, he keeps up with the job market to advise his grandchildren. Joe feels "
                          "that jobs today are more demanding in terms of education and skills than in his time. He "
                          "often reflects on how stable factory jobs have been replaced by automation and gig work, "
                          "leaving him concerned about the younger generation's ability to find secure, "
                          "lifelong careers."},
    "Agent2": {"name": "Ravi",
               "persona": "Ravi is a 35-year-old mid-career software engineer working in the tech industry, living in "
                          "a tech hub like San Francisco. He is employed but cautious due to recent layoffs in his "
                          "field and the increasing demand for specialization in AI and cloud computing. Ravi sees "
                          "opportunities in upskilling but worries about the pressure to stay ahead in a rapidly "
                          "evolving market."},
    "Agent3": {"name": "Emily",
               "persona": "Emily is a 22-year-old recent graduate in marketing living in a bustling metropolitan "
                          "area. She is currently unemployed and actively job hunting. Emily is optimistic but feels "
                          "overwhelmed by the competition and the need for internships or unpaid roles to gain "
                          "experience. She’s concerned about the scarcity of entry-level positions and the emphasis "
                          "on digital marketing skills she didn’t focus on during her studies."},
}

agent_voices = {
    "Agent1": "com.apple.eloquence.en-US.Grandpa",
    "Agent2": "com.apple.voice.compact.en-GB.Daniel",
    "Agent3": "com.apple.eloquence.en-US.Flo"
}

# LLM setup
model_id = "meta-llama/Llama-3.2-3B"
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
    new_tokens = random.randrange(50, 201, 15)
    min_tokens = 100
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Update pipeline parameters for longer output
    response = pipe(final_prompt, min_new_tokens=min_tokens, max_new_tokens= new_tokens + min_tokens,
                    stop_strings=[".", "?", "!"], return_full_text=False, tokenizer=tokenizer)
    # another variable for how long they haven't spoken

    full_text = response[0]['generated_text']


    # Extract only the generated response after the marker
    response_marker = f"{agent_name}'s response:"
    if response_marker in full_text:
        message = full_text.split(response_marker)[-1].strip()
    else:
        message = full_text.strip()

    return message


def sim_speak(text, agent, filename=""):
    """
    Speaks the given text and saves it to a new audio file.

    Args:
        text (str): The text to convert to speech.
        filename (str): Output audio file path.

    Returns:
        float: The duration of the audio in seconds.
    """
    engine = pyttsx3.init()
    engine.setProperty('voice',agent_voices[agent])
    engine.say(text)
    # engine.save_to_file(text, filename)
    engine.runAndWait()

    # Calculate duration using the generated audio file
    # with wave.open(filename, 'rb') as audio_file:
    #     frames = audio_file.getnframes()
    #     rate = audio_file.getframerate()
    #     return frames / float(rate)



# Simulate the conversation
def simulate_conversation(discussion_starter, num_agents=3, duration_minutes=2):
    conversation_history = discussion_starter

    engine = pyttsx3.init()
    engine.say(discussion_starter)
    engine.runAndWait()
    engine.stop()

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
        sim_speak(response, agent, audio_filename) # will wait appropriate time before cont.

        turn += 1  # Increment the turn count for file naming

    # after conversation, save transcript
    file_path = 'transcript.txt'
    with open(file_path, "w") as file:
        file.write(conversation_history)

    print(f"Transcript saved to {file_path}")


if __name__ == "__main__":
    simulate_conversation("What are your thoughts on the current job market?")
