# models/char_lstm.py

import torch.nn as nn

class CharLSTM(nn.Module):
     def __init__(self, vocab_size, hidden_size, num_layers):
          super().__init__()
          self.embed = nn.Embedding(vocab_size, hidden_size)
          self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
          self.fc = nn.Linear(hidden_size, vocab_size)

     def forward(self, x, hidden=None):
          x = self.embed(x)
          output, hidden = self.lstm(x, hidden)
          logits = self.fc(output)
          return logits, hidden
