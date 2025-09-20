import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
import numpy as np
from torch_geometric.nn import GCNConv

bert_features_file = 'prescriptions_bert_features.csv'
bert_features_df = pd.read_csv(bert_features_file, header=None)

prescriptions_file = 'CF_utf-8_modified.json'
with open(prescriptions_file, 'r', encoding='utf-8') as f:
    prescriptions_data = json.load(f)

# 处方数据准备
prescriptions = []
components = []
diseases = []
index_to_prescription = {}

index = 0
for prescription, info in prescriptions_data.items():
    prescriptions.append(prescription)
    components.append(info['components'][0])
    diseases.append(info['disease'])
    index_to_prescription[index] = prescription
    index += 1

# 设置节点特征和边连接
num_prescriptions = len(prescriptions)
num_features = bert_features_df.shape[1] - 1

# 准备图数据
edge_index = torch.tensor([[0, 1, 2, 3, 4],
                           [1, 0, 2, 3, 4]], dtype=torch.long)  # 无向边张量

# 节点特征 (BERT embeddings)
features = torch.tensor(bert_features_df.iloc[:, 1:].values, dtype=torch.float)

# 目标标签（target tensors for diseases）
target = torch.zeros(num_prescriptions, dtype=torch.float)
for i, d_list in enumerate(diseases):
    for d in d_list:
        if "消化不良" in d:
            target[i] = 1.0
            break

# 定义 GCN 模型
class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.classifier(x)
        return torch.sigmoid(x)

# 初始化 model
hidden_channels = 16
model = GCN(num_features=num_features, hidden_channels=hidden_channels)

# cuda加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
features = features.to(device)
edge_index = edge_index.to(device)
target = target.to(device)

# 训练设置
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# 训练 model
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(features, edge_index)
    loss = criterion(out, target.view(-1, 1))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 模型评估 (预测得分取最高输出推荐方剂)
model.eval()
with torch.no_grad():
    # 样例
    input_disease = "消化不良"

    # 查询相关方剂
    related_indices = []
    for i, d_list in enumerate(diseases):
        if any(input_disease in d for d in d_list):
            related_indices.append(i)

    # 预测得分
    prediction_scores = model(features, edge_index)
    sorted_indices = torch.argsort(prediction_scores.view(-1), descending=True)

    # 取最高得分
    top_prescription_index = -1
    for idx in sorted_indices:
        if idx.item() in related_indices:
            top_prescription_index = idx.item()
            break

    if top_prescription_index != -1:
        top_prescription = index_to_prescription[top_prescription_index]
        print(f'Top recommended prescription for "{input_disease}": {top_prescription}')
    else:
        print(f'No prescription found for "{input_disease}"')


torch.save(model.state_dict(), 'gcn_model.pth')

# 模型预测分数
prediction_scores = model(features, edge_index).cpu().detach().numpy().flatten()

# 真实标签
true_labels = torch.zeros(num_prescriptions)
for i, d_list in enumerate(diseases):
    if "消化不良" in d_list:
        true_labels[i] = 1.0

true_labels = true_labels.cpu().numpy()

# 计算混淆矩阵
threshold = 0.5  # 预测分数阈值
predicted_labels = (prediction_scores >= threshold).astype(int)

tp = np.sum((predicted_labels == 1) & (true_labels == 1))
fp = np.sum((predicted_labels == 1) & (true_labels == 0))
fn = np.sum((predicted_labels == 0) & (true_labels == 1))

accuracy = np.mean(predicted_labels == true_labels)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')