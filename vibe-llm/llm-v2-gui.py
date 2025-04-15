import tkinter as tk
from tkinter import scrolledtext
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2TokenizerFast

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

class ConversationalGUI(tk.Tk):
     def __init__(self):
          super().__init__()
          self.title("Conversational AI")
          self.geometry("600x400")
          
          # Create GUI components
          self.text_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=70, height=15, font=("Arial", 12))
          self.text_area.grid(row=0, column=0, padx=10, pady=10)
          
          self.user_input = tk.Entry(self, width=60, font=("Arial", 12))
          self.user_input.grid(row=1, column=0, padx=10, pady=10)
          
          self.send_button = tk.Button(self, text="Send", command=self.get_response, width=20, font=("Arial", 12))
          self.send_button.grid(row=2, column=0, padx=10, pady=10)
          
     def get_response(self):
          user_message = self.user_input.get()
          if user_message.lower() == "exit":
               self.quit()
          
          # Display user's message in the text area
          self.text_area.insert(tk.END, f"You: {user_message}\n")
          self.user_input.delete(0, tk.END)
          
          # Generate response from the model
          conversation_history = self.text_area.get(1.0, tk.END)
          response = generate_conversation(conversation_history)
          
          # Display the model's response in the text area
          self.text_area.insert(tk.END, f"Model: {response}\n")
          self.text_area.yview(tk.END)

if __name__ == "__main__":
     gui = ConversationalGUI()
     gui.mainloop()
