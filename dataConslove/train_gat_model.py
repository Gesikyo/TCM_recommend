import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ==== 1. Load features ====
features_df = pd.read_csv("prescriptions_bert_features.csv", header=None)
features = torch.tensor(features_df.iloc[:, 1:].values, dtype=torch.float32)
prescription_names = features_df[0].tolist()

# ==== 2. Load prescription metadata ====
with open("CF_utf-8_modified.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ==== 3. Build label matrix (multi-label disease classification) ====
disease_lists = [metadata[p]['disease'] for p in prescription_names]
mlb = MultiLabelBinarizer()
labels = torch.tensor(mlb.fit_transform(disease_lists), dtype=torch.float32)

# ==== 4. Build edges based on similarity ====
n = len(prescription_names)
edge_index = []  # [2, num_edges]

def add_edge(i, j):
    edge_index.append([i, j])
    edge_index.append([j, i])  # bidirectional

cos_sim = cosine_similarity(features.numpy())

for i in tqdm(range(n)):
    for j in range(i + 1, n):
        name_i, name_j = prescription_names[i], prescription_names[j]
        # 成分交集
        comps_i = set(metadata[name_i]['components'][0].keys())
        comps_j = set(metadata[name_j]['components'][0].keys())
        if comps_i & comps_j:
            add_edge(i, j)
        # 疾病交集
        dis_i = set(metadata[name_i]['disease'])
        dis_j = set(metadata[name_j]['disease'])
        if dis_i & dis_j:
            add_edge(i, j)
        # BERT相似度
        if cos_sim[i, j] > 0.85:
            add_edge(i, j)

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# ==== 5. Define GAT Model ====
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

# ==== 6. Build Graph Data ====
data = Data(x=features, edge_index=edge_index, y=labels)

# ==== 7. Train ====
model = GAT(features.size(1), 64, labels.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
loss_fn = BCEWithLogitsLoss()

model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ==== 8. Save Model ====
torch.save(model.state_dict(), "gat_model.pth")
print("Model saved to gat_model.pth")