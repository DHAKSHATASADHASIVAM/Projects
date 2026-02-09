
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
_api_key = os.getenv("GROQ_API_KEY")
if _api_key:
    _api_key = _api_key.strip().strip('"').strip("'")
else:
    raise RuntimeError("GROQ_API_KEY not set. Add it to your .env or environment (GROQ_API_KEY='gsk_xxx').")

client = Groq(api_key=_api_key)
completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
      {
        "role": "user",
        "content": ""
      }
    ],
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")