from openai import OpenAI
import numpy as np
import os
from flask import Flask, render_template, request

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model = "gpt-3.5-turbo-instruct"

def call_gpt(prompt):

  response = client.completions.create(model=model,
  prompt=prompt,
  max_tokens=1,
  n=1,
  stop=None,
  temperature=1.3,
  top_p=0.9,
  frequency_penalty=0,
  logprobs=5,
  presence_penalty=0.6)
  return response

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_chart_from_sentence(sentence):
  response = call_gpt(sentence)
  print(response)
  probabilities = response.choices[0].logprobs.top_logprobs[0]
# Extracting scores and normalizing probabilities
  scores = np.array(list(probabilities.values()))
  normalized_probs = softmax(scores)
  output = []
# Output probabilities for each token
  for token, probability in zip(probabilities.keys(), normalized_probs):
      # Print out the raw token, if its "\n" then don't actually print a newline
      token = token.replace("\n", "\\n")
      print(f'"{token}" with probability {probability:.2f}')
      output.append([token, round(probability, 2)])
  output.sort(key=lambda x: x[1], reverse=True)
  return output

