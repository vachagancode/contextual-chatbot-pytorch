import json
from nltk_utils import tokenizer, stem, bag_of_words
import numpy as np
import os
from model import ChatbotNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

with open("intents.json", "r") as f:
	intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
	tag = intent['tag']
	tags.append(tag)
	for pattern in intent['patterns']:
		w = tokenizer(pattern)
		all_words.extend(w)
		xy.append((w, tag))

ignore = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignore]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(all_words)
# print(tags)
# print(xy)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
	bag = bag_of_words(pattern_sentence, all_words)
	X_train.append(bag)

	label = tags.index(tag)
	y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
	def __init__(self):
		self.n_samples = len(X_train)
		self.x_data = X_train
		self.y_data = y_train

	def __getitem__(self, idx):
		return self.x_data[idx], self.y_data[idx]

	def __len__(self):
		return self.n_samples

dataset = ChatDataset()

BATCH_SIZE = 8

train_dataloader = DataLoader(
	dataset=dataset,
	batch_size=BATCH_SIZE,
	shuffle=True
)

input_size = len(X_train[0])
hidden_layers = 8
output_size = len(tags)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChatbotNN(
	input_size=input_size,
	hidden_layers=hidden_layers,
	num_classes=output_size
).to(device)

learning_rate = 0.001
num_epochs = 1000

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
	for (X, y) in train_dataloader:
		# print(X, y)
		X, y = X.to(device), y.type(torch.long).to(device)

		# do the forward pass
		y_preds = model(X)
		# print(y_preds)

		# calculate the loss
		loss = criterion(y_preds, y)

		# optimizer zero grad
		optimizer.zero_grad()

		# loss backward
		loss.backward()

		# optimizer step
		optimizer.step()

	if epoch % 100 == 0:
		print(f"Epoch: {epoch} | Loss: {loss.item():.4f}")

print(f"Final loss: {loss.item():.4f}")


data = {
	"model_state": model.state_dict(),
	"input_size": input_size,
	"hidden_layers": hidden_layers,
	"output_size": output_size,
	"all_words": all_words,
	"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f"Training completed. file saved to {FILE}")