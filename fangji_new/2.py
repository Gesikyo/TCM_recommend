# 加载模型
import torch

from fangji_new.dataJson import disease_to_bert_embedding, disease_encoder, device, GCN, data, formula_names, components

model = GCN(in_channels=2, hidden_channels=16, out_channels=len(disease_encoder.classes_)).to(device)
model.load_state_dict(torch.load('gcn_model.pth'))

def recommend_formula(user_description):
    model.eval()
    with torch.no_grad():
        user_embedding = disease_to_bert_embedding([user_description])
        user_embedding = user_embedding.repeat(len(data.x), 1)
        out = model(data)
        similarity = torch.cosine_similarity(out, user_embedding, dim=1)
        recommended_index = similarity.argmax().item()
        recommended_formula = formula_names[recommended_index]
        recommended_components = components[recommended_index]
        return recommended_formula, recommended_components

# 示例推理
user_description = "支气管哮喘，急性胰腺炎"
recommended_formula, recommended_components = recommend_formula(user_description)
print(f'Recommended Formula: {recommended_formula}')
print(f'Recommended Components: {recommended_components}')
