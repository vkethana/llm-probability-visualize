from openai import OpenAI
import numpy as np
import os
from flask import Flask, request, render_template, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from gpt2 import next_token_probabilities

USE_GPT_3 = False

if USE_GPT_3:
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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_updates(content):
    while session.get('send_updates', False):
      # Generate your JSON data here
      if USE_GPT_3:
        table_data, largest_prob_token = get_chart_from_sentence(content)
      else:
        table_data, largest_prob_token = next_token_probabilities(content)

      print(table_data, largest_prob_token)
      content += largest_prob_token
      response_data = {
        'modified_content': content,
        'table': table_data,
        'largest_prob_token': largest_prob_token
      }
      yield response_data
      time.sleep(1)  # Send update every second

@app.route('/start_updates', methods=['POST'])
def start_updates():
    session['send_updates'] = True
    session['content'] = request.form['content']
    return '', 200

@app.route('/stop_updates', methods=['POST'])
def stop_updates():
    session['send_updates'] = False
    return '', 200

@app.route('/updates')
def updates():
    content = session.get('content', "undef")
    if content == "undef":
      print("Content is undefined")
    return Response(generate_updates(content), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)

