import pandas as pd
import chardet
import os

# 指定文件夹路径
folder_path = './data'
# 获取所有文件名
file_names = [f for f in os.listdir(folder_path) if f.startswith('clear') and f.endswith('.csv')]

# 需要的列属性
required_columns = ['ID', '西医疾病', '中医疾病', '中医证候', '医案原文', '医案来源']

# 存储所有数据的列表
all_data = []

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)

    # 尝试多种编码格式
    for encoding in ['utf-8', 'gbk', 'gb2312']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except Exception as e:
            print(f"Error reading {file_path} with {encoding}: {e}")
            continue
    else:
        print(f"Failed to read {file_path} with any encoding, skipping...")
        continue

    # 保留需要的列
    df = df.reindex(columns=required_columns)

    # 添加到总数据列表中
    all_data.append(df)

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)

# 将合并后的数据保存到一个新的CSV文件中
output_file = './combined_data.csv'
combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"合并后的数据已保存到 {output_file}")
