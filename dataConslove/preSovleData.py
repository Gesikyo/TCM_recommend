import json
import pandas as pd

# 读取JSON文件
with open('CF_utf-8_modified.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取方剂信息
prescriptions = []

for formula_name, formula_info in data.items():
    components = formula_info["components"][0]
    for herb, amount in components.items():
        prescriptions.append({
            "Formula": formula_name,
            "Herb": herb,
            "Amount": amount
        })

# 将数据转换为DataFrame
df = pd.DataFrame(prescriptions)

# 保存到CSV文件
output_path = 'prescriptions_components.csv'
df.to_csv(output_path, index=False)
output_path
