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
# TODO: Automate persona generation (using number of agents). First we need to figure out what makes a good persona


personas = {
    # For + Very Related Interests
    "Agent1": {
        "name": "Sophia",
        "persona": "Sophia, a 28-year-old public policy analyst who specializes in labor rights and automation policy. She is passionate about ensuring workers have access to fair opportunities and believes innovation should align with ethical labor practices. She is a staunch supporter of policies that protect workers in the gig economy."
    },
    # For + Very Non-Related Interests
    "Agent2": {
        "name": "Carlos",
        "persona": "Carlos, a 41-year-old art historian and professor who has a deep love for Renaissance art and cultural heritage. Although his primary interests are unrelated to labor rights or gig economy issues, he supports worker advocacy because of his belief in societal fairness and cultural equity."
    },
    # Neutral + Very Related Interests
    "Agent3": {
        "name": "Mia",
        "persona": "Mia, a 34-year-old freelance UX designer who has been navigating the gig economy for the past 10 years. She sees both the positives and negatives of freelance work, from flexibility to instability. She remains neutral in conversations, as her experiences have shown her both sides of the argument."
    },
    # Neutral + Very Non-Related Interests
    "Agent4": {
        "name": "Leo",
        "persona": "Leo, a 27-year-old wildlife biologist who spends most of his time in remote field stations studying endangered species. While not personally involved in labor rights or gig economy debates, he recognizes their importance but doesn’t feel strongly for or against."
    },
    # Against + Very Related Interests
    "Agent5": {
        "name": "Ahmed",
        "persona": "Ahmed, a 50-year-old small business owner who used to employ a team but lost customers to gig economy services. He sees the gig economy as a threat to traditional businesses and believes it exploits workers under the guise of flexibility."
    },
    # Against + Very Non-Related Interests
    "Agent6": {
        "name": "Olivia",
        "persona": "Olivia, a 62-year-old retired opera singer who has little direct connection to gig economy issues. However, she strongly opposes its rise, believing it fosters a culture of disposability that undermines the arts and long-term societal investments."
    }
}


agent_voices = {
    "Agent1": "com.apple.eloquence.en-US.Grandpa",
    "Agent2": "com.apple.voice.compact.en-GB.Daniel",
    "Agent3": "com.apple.eloquence.en-US.Flo"
}

# LLM setup
model_1 = "meta-llama/Llama-3.2-1B"
pipe1 = pipeline(
    "text-generation",
    model=model_1,
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
    agent_name = personas[agent]["name"]
    agent_persona = personas[agent]["persona"]
    print('prompting', agent_name)


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
    new_tokens = random.randrange(50, 150, 25)
    min_tokens = 50
    tokenizer = AutoTokenizer.from_pretrained(model_1)
    # Update pipeline parameters for longer output
    response = pipe1(final_prompt,
                    min_new_tokens=min_tokens,
                    max_new_tokens=new_tokens + min_tokens,
                    return_full_text=False,
                    tokenizer=tokenizer,
                    pad_token_id=50256,  # Define padding token
                    no_repeat_ngram_size=2, # avoids repeats
                    temperature = 0.9,  # Slightly more randomness
                    top_p = 0.95,  # Nucleus sampling
                    stop_strings=["?"]
    )
    # stop_strings = [".", "?", "!"]
    # another variable for how long they haven't spoken

    full_text = response[0]['generated_text']
    print(full_text)


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


# todo: automate response collection for a given conversation based on a few personality traits. Then create a grid of responses and store there based on categorization


if __name__ == "__main__":
    simulate_conversation("What are your thoughts on the current job market?", duration_minutes=5)
