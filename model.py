import torch
import torch.nn as nn

class ChatbotNN(nn.Module):
	def __init__(self, input_size : int, hidden_layers : int, num_classes : int):
		super().__init__()
		self.l1 = nn.Linear(in_features=input_size, out_features=hidden_layers)
		self.l2 = nn.Linear(in_features=hidden_layers, out_features=hidden_layers)
		self.l3 = nn.Linear(in_features=hidden_layers, out_features=num_classes)
		self.relu = nn.ReLU()

	def forward(self, x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)
		out = self.relu(out)
		out = self.l3(out)

		return out