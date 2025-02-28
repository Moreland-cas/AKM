from openai import OpenAI

client = OpenAI(api_key=None) # NEED TO FILL IN YOUR API KEY LIKE 'sk-...'
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)