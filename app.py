from openai import OpenAI
import numpy as np
import os
import json
from flask import Flask, request, render_template, Response, session, stream_with_context
import time
#from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from gpt2 import next_token_probabilities

USE_GPT_3 = False

if USE_GPT_3:
  client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
  model = "gpt-3.5-turbo-instruct"

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ["FLASK_SESSION_SECRET"]
#print("FLASK_SESSION=", app.config['SECRET_KEY'])

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
  #print(response)
  probabilities = response.choices[0].logprobs.top_logprobs[0]

  # Extracting scores and normalizing probabilities
  scores = np.array(list(probabilities.values()))
  normalized_probs = softmax(scores)
  output = {}

  # Output probabilities for each token
  for token, probability in zip(probabilities.keys(), normalized_probs):
      # Print out the raw token, if its "\n" then don't actually print a newline
      token = token.replace("\n", "\\n")
      print(f'"{token}" with probability {probability:.2f}')
      output[token] = round(probability, 3)
  # Sort output by probability
  output = dict(sorted(output.items(), key=lambda item: item[1], reverse=True))
  # Get largest probability item
  largest_prob_token = max(output, key=output.get)
  print(f"Output: {output}")
  print(f"Largest probability token: {largest_prob_token}")
  return output, largest_prob_token

@app.route('/')
def index():
    return render_template('index.html')

def generate_updates(content):
    while True:
      if USE_GPT_3:
        table_data, largest_prob_token = get_chart_from_sentence(content)
      else:
        table_data, largest_prob_token = next_token_probabilities(content)

      #print(table_data, largest_prob_token)
      content += largest_prob_token

      response_data = {
        'modified_content': content,
        'table': table_data,
        'largest_prob_token': largest_prob_token
      }
      #yield json.dumps(response_data)
      yield f"data: {json.dumps(response_data)}\n\n"
      time.sleep(0.1)  # Send update every second

@app.route('/update_sentence', methods=['POST'])
def update_sentence():
    with app.app_context():
      session['content'] = str(request.get_json())
      print("JSON: ", session['content'])
    return "Successfully updated", 200
    #return render_template('index.html')

@app.route('/updates')
def updates():
    content = None
    with app.app_context():
      content = session.get('content', "...")
    print("retreived sentence content", content)
    return Response(generate_updates(content), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
