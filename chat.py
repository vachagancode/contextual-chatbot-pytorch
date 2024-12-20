import random
import json
import torch
from model import ChatbotNN
from nltk_utils import bag_of_words, tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r") as f:
	intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE, weights_only=True)
# print(data)
input_size = data["input_size"]
output_size = data["output_size"]
hidden_layers = data["hidden_layers"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model = ChatbotNN(
	input_size=input_size,
	hidden_layers=hidden_layers,
	num_classes=output_size
).to(device)

model.load_state_dict(model_state)

model.eval()

print(f"You connected to an AI Chatbot for a moped store.")
print("--------------------------------------------------")
bot_name = "Bot"
print(f"Let's chat: type 'quit' to exit")
while True:
	sentence = input("You: ")
	if sentence == "quit":
		break

	sentence = tokenizer(sentence)
	X = bag_of_words(sentence, all_words)
	X =  X.reshape(1, X.shape[0])
	X = torch.from_numpy(X)

	output = model(X)
	prediction, class_tag = torch.max(output, dim=1)
	tag = tags[class_tag.item()]

	probs = torch.softmax(output, dim=1)
	probability = int(probs[0][class_tag.item()] * 100)

	if probability > 75:
		for intent in intents["intents"]:
			if tag == intent["tag"]:
				response = random.choice(intent["responses"])
				print(f"{bot_name}: {response}")
	else:
		print(f"{bot_name}: Sorry, I didn't get it...")