# train.py

import torch
from torch import nn, optim
from models.char_lstm import CharLSTM
from utils import load_data, encode, get_batches
from config import *

# Load and prepare data
text, stoi, itos = load_data('data/input.txt')
encoded_text = encode(text, stoi)
vocab_size = len(stoi)

# Init model
model = CharLSTM(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(EPOCHS):
     total_loss = 0
     for inputs, targets in get_batches(encoded_text, SEQ_LENGTH, BATCH_SIZE):
          inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
          optimizer.zero_grad()
          logits, _ = model(inputs)
          loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
          loss.backward()
          optimizer.step()
          total_loss += loss.item()
     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Save the model
torch.save({
     'model': model.state_dict(),
     'stoi': stoi,
     'itos': itos
}, 'char_lstm.pth')
