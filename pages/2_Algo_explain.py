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

# from elevenlabs import Voice, VoiceSettings, generate, set_api_key

import tiktoken
st.set_page_config(layout="wide")
# image = Image.open('algo_inwork.png')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

gpt_model= st.sidebar.selectbox("Set GPT model",('gpt-4','gpt-3.5-turbo'),index=1)

with st.sidebar:
    st.write("""| Model                   | Input Cost per 1K Tokens | Output Cost per 1K Tokens |
|-------------------------|--------------------------|---------------------------|
| gpt-4                   | $0.030                  | $0.060                     |
| gpt-3.5-turbo           | $0.001                  | $0.002                     |
             
""")

left_co, right_co = st.columns(2)
with left_co:
    st.image('algo_inwork.png', caption='ALGO', width=300)
with right_co:
    st.markdown("""*This page is under construction. It will be a place to copy and paste code for general feedback or explanation*""")


def main():
    stuff='stuff'

if __name__ == "__main__":
    if not openai_api_key.startswith('sk-'):
        st.info('Please enter your OpenAI API key on the left sidebar. You can create one at https://openai.com/blog/openai-api', icon='⚠')
    else:
        main()