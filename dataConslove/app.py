from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
import json
from torch_geometric.nn import GCNConv
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载模型和数据
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

# 加载数据
bert_features_file = 'prescriptions_bert_features.csv'
bert_features_df = pd.read_csv(bert_features_file, header=None)
num_features = bert_features_df.shape[1] - 1  # 排除方剂名称列

prescriptions_file = 'CF_utf-8_modified.json'
with open(prescriptions_file, 'r', encoding='utf-8') as f:
    prescriptions_data = json.load(f)

prescriptions = []
diseases = []
index_to_prescription = {}

index = 0
for prescription, info in prescriptions_data.items():
    prescriptions.append(prescription)
    diseases.append(info['disease'])
    index_to_prescription[index] = {
        'name': prescription,
        'components': info['components'],
        'disease': info['disease']
    }
    index += 1

# 准备图数据
edge_index = torch.tensor([[0, 1, 2, 3, 4],
                           [1, 0, 2, 3, 4]], dtype=torch.long)  # 无向边
features = torch.tensor(bert_features_df.iloc[:, 1:].values, dtype=torch.float)

# 加载预训练模型
hidden_channels = 16
model = GCN(num_features=num_features, hidden_channels=hidden_channels)
model.load_state_dict(torch.load('gcn_model.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
features = features.to(device)
edge_index = edge_index.to(device)

@app.route('/recommend', methods=['POST'])
def recommend():
    input_data = request.json
    input_disease = input_data.get('disease', '')
    input_symptoms = input_disease.split()

    # 查找与输入疾病相关的方剂
    related_indices = []
    for i, d_list in enumerate(diseases):
        if any(symptom in d_list for symptom in input_symptoms):
            related_indices.append(i)

    # 预测相关方剂的分数
    with torch.no_grad():
        prediction_scores = model(features, edge_index)
        sorted_indices = torch.argsort(prediction_scores.view(-1), descending=True)

        # 获取推荐的前3个方剂
        top_prescriptions = []
        for idx in sorted_indices:
            if idx.item() in related_indices:
                top_prescriptions.append(index_to_prescription[idx.item()])
                if len(top_prescriptions) == 3:
                    break

        if top_prescriptions:
            return jsonify(top_prescriptions)
        else:
            return jsonify({'prescription': 'No recommendation found'}), 404

if __name__ == '__main__':
    app.run(debug=True)

