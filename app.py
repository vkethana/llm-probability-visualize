from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model = "gpt-3.5-turbo-instruct"

def call_gpt(prompt):
  '''
  Call GPT for the beamsearch algorithm
  '''

  response = client.completions.create(model=model,
  prompt=prompt,
  max_tokens=20,
  n=1,
  stop=None,
  temperature=1.3,
  top_p=0.9,
  frequency_penalty=0,
  logprobs=1,
  top_logprobs=5,
  presence_penalty=0.6)
  return response

print(call_gpt("Once upon a time"))
