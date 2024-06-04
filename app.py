from openai import OpenAI
import numpy as np
import os
import json
from flask import Flask, request, render_template, Response, session, stream_with_context, jsonify
from flask_session import Session
import time
from gpt2 import next_token_probabilities as next_token_probs_gpt_2

'''
In order to run this app on a webserver, you need to make an environment variable for the Flask session secret
'''
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ["FLASK_SESSION_SECRET"]
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

def call_gpt(prompt, client):
  '''
  This method is only used for gpt3
  '''
  response = client.completions.create(model='gpt-3.5-turbo-instruct',
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

def next_token_probs_gpt_3(sentence, api_key):
  '''
  Gpt-2 inference is located in a separate function.
  '''
  client = OpenAI(api_key=api_key)
  try:
    response = call_gpt(sentence, client)
  except Exception as e:
    return {}, ""

  probabilities = response.choices[0].logprobs.top_logprobs[0]

  # Extracting scores and normalizing probabilities
  scores = np.array(list(probabilities.values()))
  normalized_probs = softmax(scores)
  output = {}

  # Output probabilities for each token
  for token, probability in zip(probabilities.keys(), normalized_probs):
      # Print out the raw token, if its "\n" then don't actually print a newline
      token = token.replace("\n", "\\n")
      #print(f'"{token}" with probability {probability:.2f}')
      output[token] = round(probability, 3)
  # Sort output by probability
  output = dict(sorted(output.items(), key=lambda item: item[1], reverse=True))
  # Get largest probability item
  largest_prob_token = max(output, key=output.get)
  #print(f"Output: {output}")
  #print(f"Largest probability token: {largest_prob_token}")
  return output, largest_prob_token

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extend_sentence', methods=['POST'])
def generate_updates():
    # Get the content from the request
    data = request.get_json()
    content = data['content']
    model = data['model']
    api_key = data['api_key']

    #print(f"Content: {content}")
    if content == None or content == "":
      #print("Blank content")
      return "data: {}\n\n"

    if model == 'gpt-3':
      table_data, largest_prob_token = next_token_probs_gpt_3(content, api_key)
    else:
      table_data, largest_prob_token = next_token_probs_gpt_2(content)

    content += largest_prob_token

    response_data = {
      'modified_content': content,
      'table': table_data,
      'largest_prob_token': largest_prob_token
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
