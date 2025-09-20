import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 加载特征数据
df = pd.read_csv('prescriptions_bert_features.csv', header=None, index_col=0)

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# 使用PCA降维
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# 创建可视化DataFrame
vis_df = pd.DataFrame({
    'Prescription': df.index,
    'PC1': pca_features[:, 0],
    'PC2': pca_features[:, 1]
})

# 设置绘图样式
plt.figure(figsize=(12, 10))
sns.set(style="whitegrid")

# 创建散点图
scatter = sns.scatterplot(
    x='PC1',
    y='PC2',
    data=vis_df,
    alpha=0.6,
    edgecolor='w',
    s=80
)

plt.title('BERT Features Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False # 设置正常显示符号

# 可选：随机标注部分方剂名称（避免过于拥挤）
for i in range(0, len(vis_df), 20):  # 每20个标注一个
    scatter.text(
        vis_df.PC1[i] + 0.02,
        vis_df.PC2[i] + 0.02,
        vis_df.Prescription[i],
        horizontalalignment='left',
        size='small',
        color='black'
    )

plt.savefig('bert_features_pca.png', dpi=300, bbox_inches='tight')
plt.show()