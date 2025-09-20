import json

with open('pre_list.json', 'r', encoding='ascii') as f:
    pre_list = json.load(f)

with open('herb_list.json', 'r', encoding='ascii') as f:
    herb_list = json.load(f)
herb_dict = {herb[1]: i for i, herb in enumerate(herb_list)}
effect_dict = {}
effect_idx = 0
for pre in pre_list:
    for effect in pre[4]:
        if effect not in effect_dict:
            effect_dict[effect] = effect_idx
            effect_idx += 1
import numpy as np


def create_feature_and_label(pre_list, herb_dict, effect_dict):
    num_herbs = len(herb_dict)
    num_effects = len(effect_dict)

    features = []
    labels = []

    for pre in pre_list:
        feature = np.zeros(num_herbs)
        for herb, amount in zip(pre[2], pre[3]):
            feature[herb_dict[herb]] = float(amount)
        label = np.zeros(num_effects)
        for effect in pre[4]:
            label[effect_dict[effect]] = 1
        features.append(feature)
        labels.append(label)

    return np.array(features), np.array(labels)


X, y = create_feature_and_label(pre_list, herb_dict, effect_dict)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
batch_size = 4

dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class HerbModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HerbModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


input_size = X.shape[1]
hidden_size = 64
output_size = y.shape[1]

model = HerbModel(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    for batch_features, batch_labels in dataloader:
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
def predict(model, herbs, herb_dict):
    feature = np.zeros(len(herb_dict))
    for herb, amount in herbs:
        feature[herb_dict[herb]] = float(amount)
    feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
    output = model(feature)
    predicted = (output > 0.5).float()
    return predicted

# 示例预测
herbs = [("厚朴", 15), ("甘草", 9), ("大黄", 9), ("大枣", 20), ("枳实", 9), ("桂枝", 6), ("生姜", 12)]
predicted = predict(model, herbs, herb_dict)
print(predicted)
