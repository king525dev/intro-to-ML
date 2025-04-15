# utils.py

import torch

def load_data(file_path):
     with open(file_path, 'r', encoding='utf-8') as f:
          text = f.read()
     chars = sorted(list(set(text)))
     stoi = {ch: i for i, ch in enumerate(chars)}
     itos = {i: ch for ch, i in stoi.items()}
     return text, stoi, itos

def encode(text, stoi):
     return [stoi[ch] for ch in text]

def decode(indices, itos):
     return ''.join([itos[i] for i in indices])

def get_batches(data, seq_length, batch_size):
     total_length = len(data) - seq_length
     for i in range(0, total_length, seq_length * batch_size):
          inputs = []
          targets = []
          for j in range(batch_size):
               start = i + j * seq_length
               end = start + seq_length
               if end + 1 >= len(data):
                    break
               inputs.append(data[start:end])
               targets.append(data[start+1:end+1])
          yield torch.tensor(inputs), torch.tensor(targets)
