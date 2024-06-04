import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch.nn.functional as F

# Load pre-trained model and tokenizer
model_name = 'gpt2'  # You can use 'gpt2-medium', 'gpt2-large', or 'gpt2-xl' as well
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def next_token_probabilities(prompt):
    # Encode the input text
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Get the logits from the model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Get the logits for the last token
    last_token_logits = logits[0, -1, :]

    # Apply softmax to get probabilities
    probabilities = F.softmax(last_token_logits, dim=-1)

    # Get the top 10 most probable next tokens
    top_k = 10
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

    # Normalize the probabilities so they sum to 1
    top_k_probs = top_k_probs / top_k_probs.sum()

    # Convert the indices to tokens
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]

    words = list(zip(top_k_tokens, top_k_probs.cpu().numpy()))
    # Get largest word in words
    max_word = max(words, key=lambda x: x[1])[0]
    #print("Biggest word:", max_word)
    my_dict = {}
    for word, prob in words:
        my_dict[word] = round(100 * float(prob), 3)
    return my_dict, max_word

if __name__ == "__main__":
  # example usage
  prompt = "the quick brown fox"
  top_k_predictions, max_word = next_token_probabilities(prompt)
  #print("top predictions:", top_k_predictions, "\nmax word:", max_word)
