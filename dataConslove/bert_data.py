import json
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# 加载CF_utf-8_modified.json文件
with open('CF_utf-8_modified.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化Bert tokenizer和model
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 提取方剂名称和成分
prescriptions = list(data.keys())
components = [list(data[prescription]['components'][0].keys()) for prescription in prescriptions]

# 将成分列表转换为文本格式
component_texts = [' '.join(comp) for comp in components]

# 使用Bert对成分进行特征化
inputs = tokenizer(component_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
features = outputs.last_hidden_state[:, 0, :].numpy()

# 创建DataFrame并保存为CSV
df = pd.DataFrame(features, index=prescriptions)
df.to_csv('prescriptions_bert_features.csv', header=False)

# 创建方剂到索引的映射并保存为json文件
prescription_index = {prescription: idx for idx, prescription in enumerate(prescriptions)}
with open('prescription_index.json', 'w', encoding='utf-8') as f:
    json.dump(prescription_index, f, ensure_ascii=False)
