import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# 加载数据
with open('CF_utf-8_modified.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有中药名称
all_herbs = set()
for prescription, details in data.items():
    for component_dict in details['components']:
        for herb in component_dict.keys():
            all_herbs.add(herb)

# 转换为DataFrame格式
prescriptions = []
herbs_data = []

for prescription, details in data.items():
    for component_dict in details['components']:
        herb_dict = {herb: 0 for herb in all_herbs}
        for herb, amount in component_dict.items():
            herb_dict[herb] = amount
        herbs_data.append(herb_dict)
        prescriptions.append(prescription)

df = pd.DataFrame(herbs_data, index=prescriptions)

# 处理重复的处方名（如果有多个相同处方的不同组件）
df = df.groupby(df.index).sum()

# 为热力图准备数据
# 只选择示例中出现的三个处方
sample_prescriptions = ["厚朴七物汤", "桂枝人参汤", "乌头桂枝汤"]
df_sample = df.loc[sample_prescriptions]

# 移除全为0的列（未在这三个处方中使用的药材）
df_sample = df_sample.loc[:, (df_sample != 0).any(axis=0)]

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建热力图
plt.figure(figsize=(14, 8))
sns.heatmap(df_sample, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
plt.title('中药方剂药材用量热力图', fontsize=18)
plt.xlabel('药材', fontsize=14)
plt.ylabel('方剂', fontsize=14)

# 调整布局
plt.tight_layout()
plt.savefig('chinese_medicine_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 扩展：按药材使用频率创建另一个热力图
herb_freq = {}
for herb in all_herbs:
    count = 0
    for prescription, details in data.items():
        for component_dict in details['components']:
            if herb in component_dict and component_dict[herb] > 0:
                count += 1
    herb_freq[herb] = count

# 选择前15种最常用的药材
top_herbs = dict(sorted(herb_freq.items(), key=lambda x: x[1], reverse=True)[:15])

# 创建包含所有处方的数据框
all_prescriptions_df = df.copy()

# 只保留最常用的药材列
common_herbs_df = all_prescriptions_df[[herb for herb in top_herbs.keys() if herb in all_prescriptions_df.columns]]

# 按用药数量降序排列处方
prescription_herb_count = common_herbs_df.astype(bool).sum(axis=1)
common_herbs_df = common_herbs_df.loc[prescription_herb_count.sort_values(ascending=False).index]

# 只选择前10个处方（用药种类最多的）
top_prescriptions_df = common_herbs_df.iloc[:10]

# 创建第二个热力图 - 最常用药材在常用处方中的用量
plt.figure(figsize=(16, 10))
sns.heatmap(top_prescriptions_df, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5)
plt.title('常用中药在主要方剂中的用量热力图', fontsize=18)
plt.xlabel('常用药材', fontsize=14)
plt.ylabel('主要方剂', fontsize=14)

# 调整布局
plt.tight_layout()
plt.savefig('common_herbs_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()