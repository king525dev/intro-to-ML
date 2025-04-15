import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import argparse
import tkinter as tk
from tkinter import scrolledtext

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Multi-level dataset fallback
def load_text_dataset():
    dataset_order = [
        ("daily_dialog", lambda ds: " ".join(ds["train"][0]["dialog"])),
        ("conv_ai_2", lambda ds: " ".join([utter["text"] for utter in ds["train"][0]["dialog"]])),
        ("cornell_movie_dialog", lambda ds: ds["train"][0]["utterance"]),
        ("empathetic_dialogues", lambda ds: ds["train"][0]["utterance"] + " " + ds["train"][0]["context"]),
        ("knkarthick/dialogsum", lambda ds: ds["train"][0]["dialogue"]),
        ("multi_woz_v22", lambda ds: " ".join([turn["text"] for turn in ds["train"][0]["dialogue"]])),
        ("mctaco", lambda ds: ds["train"][0]["question"] + " " + ds["train"][0]["answer"]),
        ("wikihow", lambda ds: ds["train"][0]["headline"] + " " + ds["train"][0]["text"]),
        ("hellaswag", lambda ds: ds["train"][0]["ctx"] + " " + ds["train"][0]["endings"][0]),
    ]

    for dataset_name, extractor in dataset_order:
        try:
            print(f"🔍 Trying to load dataset: '{dataset_name}'...")
            ds = load_dataset(dataset_name)
            text = extractor(ds)
            print(f"✅ Successfully loaded '{dataset_name}'")
            return text
        except Exception as e:
            print(f"⚠️ Failed to load '{dataset_name}': {e}")

    # Final fallback to tiny_shakespeare
    try:
        print("🧾 Falling back to 'tiny_shakespeare' dataset...")
        ds = load_dataset("tiny_shakespeare")
        text = ds["train"][0]["text"]
        print("✅ Loaded 'tiny_shakespeare'")
        return text
    except Exception as e:
        print("❌ All dataset loading attempts failed.")
        raise RuntimeError("No datasets could be loaded.") from e

# Load data and tokenize
text = load_text_dataset()
tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
input_ids = tokens["input_ids"]

# Define MiniGPT
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

# Initialize model
model = MiniGPT(vocab_size=tokenizer.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()

BATCH_SIZE = 2
SEQ_LEN = 128

def get_batch():
    i = torch.randint(0, input_ids.size(1) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([input_ids[0, j:j+SEQ_LEN] for j in i])
    y = torch.stack([input_ids[0, j+1:j+SEQ_LEN+1] for j in i])
    return x, y

# Training (quick loop for testing)
for step in range(200):  # reduce for speed
    x, y = get_batch()
    logits = model(x)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Generation
def generate_conversation(prompt, max_new_tokens=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0])

# CLI Mode
def cli_interface():
    print("🧠 MiniGPT Chat — Type 'exit' to quit.")
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

# GUI Mode
class ConversationalGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MiniGPT Chat")
        self.geometry("600x400")

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

        self.text_area.insert(tk.END, f"You: {user_message}\n")
        self.user_input.delete(0, tk.END)

        conversation_history = self.text_area.get(1.0, tk.END)
        response = generate_conversation(conversation_history)
        response = response[len(conversation_history):]
        self.text_area.insert(tk.END, f"Model: {response}\n")
        self.text_area.yview(tk.END)

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGPT Chat Interface")
    parser.add_argument("--gui", action="store_true", help="Launch desktop GUI")
    parser.add_argument("--cli", action="store_true", help="Start CLI chat")
    args = parser.parse_args()

    if args.gui:
        app = ConversationalGUI()
        app.mainloop()
    elif args.cli:
        cli_interface()
    else:
        print("👉 Please run with --cli or --gui")
