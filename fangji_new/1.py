import json
import torch
from transformers import BertTokenizer, BertModel

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取JSON文件
with open('CF_utf-8_modified.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理数据
processed_data = []

for formula_name, info in data.items():
    components_list = info['components']  # 获取成分列表（可能有多个成分列表）
    diseases = info['disease']

    for components in components_list:
        # 处理每个成分列表
        feature_vector = []
        for component, amount in components.items():
            # 这里可以根据需求进行BERT tokenization或者其他特征处理
            feature_vector.append((component, amount))

        # 处理疾病描述，使用BERT模型获取描述的上下文表示
        # 假设取第一个疾病描述
        disease_description = diseases[0]
        inputs = tokenizer(disease_description, return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            pooled_output = outputs.pooler_output  # 获取池化后的输出作为特征向量

        # 将BERT表示和成分特征向量整合成一个综合的特征向量
        integrated_feature_vector = {
            'formula_name': formula_name,
            'feature_vector': feature_vector,
            'disease_description': disease_description,
            'bert_features': pooled_output.tolist()  # 将Tensor转换为列表形式
        }

        processed_data.append(integrated_feature_vector)

# 打印示例
for item in processed_data[:5]:  # 打印前五个方剂的示例
    print("方剂名称:", item['formula_name'])
    print("成分特征向量:", item['feature_vector'])
    print("疾病描述:", item['disease_description'])
    print("BERT特征向量:", item['bert_features'])
    print()
