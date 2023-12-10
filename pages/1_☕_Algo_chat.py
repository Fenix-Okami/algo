import streamlit as st
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
import os
from io import BytesIO
from pydub import AudioSegment
import numpy as np
from datetime import datetime
import pytz
import regex as re
# import other required libraries

from streamlit.components.v1 import html
from PIL import Image

from elevenlabs import Voice, VoiceSettings, generate, set_api_key

import tiktoken
st.set_page_config(layout="wide")
image = Image.open('algo.png')
response_template= """Persona: You are ALGO, A helpful assistant for teaching Python programming and general programming concepts.

Detailed description: ALGO, the Algorithmic Learning Guide Online, will be equipped with personalization features to enhance its teaching effectiveness. It will recognize and adapt to repeated questions, providing additional explanations or alternative perspectives to aid understanding. ALGO will also be attentive to signs that the user is struggling with a concept, offering tailored support, simplified explanations, or suggesting different learning approaches. This personalized approach will ensure that each user's learning needs are met, making the educational experience more effective and enjoyable.
"""

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
voice_toggle = st.sidebar.toggle('Generate audio (additional fees apply)',value=False) 
image_toggle = st.sidebar.toggle('Generate illustrations (additional fees apply)',value=False) 

gpt_model= st.sidebar.selectbox("Set GPT model",('gpt-4','gpt-3.5-turbo'),index=1)

reset_button = st.sidebar.button("Reset Chat")

with st.sidebar:
    st.write("""| Model                   | Input Cost per 1K Tokens | Output Cost per 1K Tokens |
|-------------------------|--------------------------|---------------------------|
| gpt-4                   | $0.030                  | $0.060                     |
| gpt-3.5-turbo           | $0.001                  | $0.002                     |
             
""")

left_co, right_co = st.columns(2)
with left_co:
    st.image('algo.png', caption='ALGO', width=300)
with right_co:
    st.markdown("""*placeholder*""")

central_timezone = pytz.timezone('America/Chicago')
current_time = datetime.now(central_timezone).time()

# Determine the part of the day
if current_time.hour < 12:
    greeting =  "Good morning!"
elif 12 <= current_time.hour < 18:
    greeting =  "Good afternoon!"
else:
    greeting = "Good evening!"

greeting=f"""{greeting} How may I help you?"""

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

if 'prompt_tokens' not in locals():
    prompt_tokens=0
    generated_tokens=0
    token_usage=0

if openai_api_key.startswith('sk-'):
    chat_model = ChatOpenAI(openai_api_key = openai_api_key)
    client = OpenAI(api_key = openai_api_key)

def get_image(prompt):
     gpt_response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt="""the following is the AI response from a chat. 
            Take this prompt and create a new prompt, in english, to pass
            to the DALL-E-3 text-to-image generator to create an appropriate illustration.
            : """+ prompt
            )
     image_prompt=gpt_response.choices[0].text
     response = client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                )
     return response.data[0].url

def get_audio(message):

    if False:
        audio = generate(
        text= message,
        voice=Voice(
            voice_id='tXeQxRt68UoVE3EoscDD',
            settings=VoiceSettings(stability=0.30, similarity_boost=0.75, style=0.0, use_speaker_boost=True)
                ),
            model="eleven_multilingual_v2"
        )
        return audio
 
    else:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="echo",
            input=message,
            response_format="mp3"
        )
        response.stream_to_file("output.mp3")
        audio_file = open('output.mp3', 'rb')
        audio_bytes = audio_file.read()

        # st.audio(audio_bytes, format='audio/mp3')
        return audio_bytes 

   
def main():
    st.title("ALGO")

    if reset_button:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.markdown('')
 
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = gpt_model

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Algo"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        system=[{"role":"system","content":response_template.format()}]
        with st.chat_message("assistant",avatar=np.array(image)):
            message_placeholder = st.empty()
            full_response = ""

            for response in client.chat.completions.create(
                                                        model=st.session_state["openai_model"],
                                                        messages=system+[
                                                            {"role": m["role"], "content": m["content"]}
                                                            for m in st.session_state.messages],
                                                        stream=True,
                                                        ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
                
            ###Track token usage for awareness
            messages =          system+[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            prompt_tokens =     num_tokens_from_messages(messages,gpt_model)
            generated_tokens  = num_tokens_from_string(full_response,'cl100k_base')
            token_usage =       prompt_tokens + generated_tokens

            message_placeholder.markdown(full_response)

            if voice_toggle:
                with st.spinner('Generating Audio...'):
                    st.audio(get_audio(full_response), format='audio/mp3')

            if image_toggle:
                with st.spinner('Generating illustration...'):
                    st.image(get_image(full_response))
        st.session_state.messages.append({"role": "assistant", "content": full_response,"avatar":np.array(image)})

        with st.sidebar:
            cost=prompt_tokens/1000*.001+generated_tokens/1000*.002
            max_context=4096
            if gpt_model=='gpt-4':
                cost=cost*30
                max_context=max_context*2
            st.code(f"""
                    --last response--
                    input: {prompt_tokens} tokens
                    output: {generated_tokens} tokens
                    estimated cost: ${cost:.5f} dollars
                    {token_usage}/{max_context} tokens in full context
                    """)

if __name__ == "__main__":
    if not openai_api_key.startswith('sk-'):
        st.info('Please enter your OpenAI API key on the left sidebar. You can create one at https://openai.com/blog/openai-api', icon='⚠')
    else:
        main()