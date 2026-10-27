
import os
import time
import logging

import streamlit as st
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing.")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=30_000,
        retry_options=types.HttpRetryOptions(
            attempts=1,
        ),
    ),
)

start = time.perf_counter()

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with only the word OK.",
        config=types.GenerateContentConfig(
            max_output_tokens=32,
        ),
    )

    print(f"API response time: {time.perf_counter() - start:.2f}s")
    print("Response:", response.text)

except Exception as e:
    print(f"Request failed after {time.perf_counter() - start:.2f}s")
    print("Error type:", type(e).__name__)
    print("Error:", str(e)[:1000])

finally:
    client.close()
