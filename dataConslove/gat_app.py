import gradio as gr
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
features_df = pd.read_csv("prescriptions_bert_features.csv", header=None)
features = torch.tensor(features_df.iloc[:, 1:].values, dtype=torch.float32)
prescription_names = features_df[0].tolist()

with open("CF_utf-8_modified.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

disease_lists = [metadata[p]['disease'] for p in prescription_names]
mlb = MultiLabelBinarizer()
labels = torch.tensor(mlb.fit_transform(disease_lists), dtype=torch.float32)

# Graph edges (match training logic)
edge_index = []
cos_sim = cosine_similarity(features.numpy())
for i in range(len(prescription_names)):
    for j in range(i + 1, len(prescription_names)):
        name_i, name_j = prescription_names[i], prescription_names[j]
        comps_i = set(metadata[name_i]['components'][0].keys())
        comps_j = set(metadata[name_j]['components'][0].keys())
        if comps_i & comps_j or set(metadata[name_i]['disease']) & set(metadata[name_j]['disease']) or cos_sim[i, j] > 0.85:
            edge_index.append([i, j])
            edge_index.append([j, i])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

data = Data(x=features, edge_index=edge_index)

# Load model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

model = GAT(features.size(1), 64, labels.size(1))
model.load_state_dict(torch.load("gat_model.pth", map_location="cpu"))
model.eval()

# Gradio interface
def recommend(symptom_input):
    user_symptoms = symptom_input.strip().split()
    synonyms = {"发烧": "发热", "咳嗽": "咳"}
    user_symptoms = [synonyms.get(s, s) for s in user_symptoms]

    with torch.no_grad():
        preds = torch.sigmoid(model(data.x, data.edge_index))

    scores = []
    for i, name in enumerate(prescription_names):
        matched = [s for s in user_symptoms if any(s in d for d in metadata[name]['disease'])]
        score = len(matched) / len(user_symptoms)
        scores.append((score, i, matched))

    top_scores = sorted(scores, key=lambda x: x[0], reverse=True)[:3]

    output = ""
    for score, idx, matched in top_scores:
        name = prescription_names[idx]
        components = metadata[name]['components'][0]
        adjusted = {
            herb: round(dose * (1 + score), 2) for herb, dose in components.items()
        }
        output += f"\n{name}\n"
        output += "成分与用量：\n"
        output += "- " + ", ".join([f"{herb}: {dose}g" for herb, dose in adjusted.items()]) + "\n"
        output += "\n功效主治：\n"
        output += "- " + "\n- ".join(metadata[name]['disease']) + "\n"
        output += f"\n推荐理由：匹配症状: {'、'.join(matched)}，匹配度: {score:.2f}\n"
    return output.strip()

demo = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(label="请输入症状（空格分隔）"),
    outputs=gr.Textbox(label="推荐结果")
)

demo.launch(share=True)

