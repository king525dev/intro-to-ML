# generate.py

import torch
from models.char_lstm import CharLSTM
from utils import decode
from config import *

def load_model():
     checkpoint = torch.load('char_lstm.pth', map_location=DEVICE)
     stoi, itos = checkpoint['stoi'], checkpoint['itos']
     vocab_size = len(stoi)
     model = CharLSTM(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
     model.load_state_dict(checkpoint['model'])
     model.eval()
     return model, stoi, itos

def generate_text(prompt, length=200):
     model, stoi, itos = load_model()
     input_seq = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long).to(DEVICE)
     hidden = None
     generated = list(prompt)

     for _ in range(length):
          logits, hidden = model(input_seq, hidden)
          next_char_logits = logits[0, -1]
          prob = torch.softmax(next_char_logits, dim=0).detach()
          next_char = torch.multinomial(prob, 1).item()
          generated.append(itos[next_char])
          input_seq = torch.tensor([[next_char]], dtype=torch.long).to(DEVICE)

     return ''.join(generated)

# Example usage
print(generate_text("Once upon a time "))
