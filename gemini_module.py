import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import warnings
warnings.filterwarnings('ignore')

import google.generativeai as genai
from pathlib import Path

api_key = os.getenv('GOOGLE_API_KEY')
# api_key = "API_KEY"

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")

chain_cache = None

def load_gemini(api_key):
    global chain_cache
    if chain_cache is None:
        genai.configure(api_key=api_key)
        chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-2.5-flash", max_output_tokens=300)

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_messages([
            ("system", "your system prompt..."),
            ("user", "{input}")
        ])

        output_parser = StrOutputParser()

        chain_cache = prompt | chat_model | output_parser
    return chain_cache

def gemini_get_response(input_text: str) -> str:
    if not input_text or not input_text.strip():
        input_text = "Hello, what can you tell me?"

    chain = load_gemini(api_key)
    response = chain.invoke({"input": input_text})

    return response