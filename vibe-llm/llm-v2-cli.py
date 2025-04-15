import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import argparse

# Load a conversational dataset (Persona-Chat)
dataset = load_dataset("persona_chat", trust_remote_code=True)
conversation = dataset['train'][0]['dialog']  # Get a conversation from the dataset

# Tokenize the Conversation
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Join the dialogue into a single prompt (for simplicity)
text = " ".join(conversation)
tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
input_ids = tokens["input_ids"]

# Define the Mini GPT Model
class MiniGPT(nn.Module):
     def __init__(self, vocab_size, dim=256, heads=4, depth=4, seq_len=128):
          super().__init__()
          self.token_embed = nn.Embedding(vocab_size, dim)
          self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
          self.blocks = nn.Sequential(*[
               nn.TransformerEncoderLayer(d_model=dim, nhead=heads) for _ in range(depth)
          ])
          self.norm = nn.LayerNorm(dim)
          self.head = nn.Linear(dim, vocab_size)

     def forward(self, x):
          b, t = x.size()
          x = self.token_embed(x) + self.pos_embed[:, :t, :]
          x = self.blocks(x)
          x = self.norm(x)
          return self.head(x)

# Create Training Loop
model = MiniGPT(vocab_size=tokenizer.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()

BATCH_SIZE = 2
SEQ_LEN = 128

def get_batch():
     i = torch.randint(0, input_ids.size(1) - SEQ_LEN, (BATCH_SIZE,))
     x = torch.stack([input_ids[0, j:j+SEQ_LEN] for j in i])
     y = torch.stack([input_ids[0, j+1:j+SEQ_LEN+1] for j in i])
     return x, y

# Training Loop
for step in range(1000):
     x, y = get_batch()
     logits = model(x)
     loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
     loss.backward()
     optimizer.step()
     optimizer.zero_grad()
     if step % 100 == 0:
          print(f"Step {step}, Loss: {loss.item():.4f}")

# Generate Conversational Responses
def generate_conversation(prompt, max_new_tokens=50):
     model.eval()
     input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
     for _ in range(max_new_tokens):
          logits = model(input_ids)
          next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
          input_ids = torch.cat([input_ids, next_token], dim=1)
     return tokenizer.decode(input_ids[0])

# CLI for interaction
def cli_interface():
     parser = argparse.ArgumentParser(description="Conversational Model CLI")
     parser.add_argument('--start', action='store_true', help="Start a conversation")
     args = parser.parse_args()

     if args.start:
          print("Welcome to the conversational model! Type 'exit' to end the conversation.")
          conversation_history = ""
          while True:
               user_input = input("You: ")
               if user_input.lower() == "exit":
                    print("Goodbye!")
                    break
               conversation_history += f"You: {user_input}\n"
               response = generate_conversation(conversation_history)
               response = response[len(conversation_history):]
               conversation_history += f"Model: {response}\n"
               print(f"Model: {response}")

if __name__ == "__main__":
     cli_interface()


"""
Usage:
` python llm-v2-cli.py --start `
"""