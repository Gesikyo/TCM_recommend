import pandas as pd

# 读取合并后的数据文件
combined_df = pd.read_csv('./combined_data.csv')

# 删除ID不存在的行
combined_df.dropna(subset=['ID'], inplace=True)

# 保存修改后的数据文件
combined_df.to_csv('./combined_data_without_missing_ID.csv', index=False)

print("已删除ID不存在的行的数据已保存到 combined_data_without_missing_ID.csv 文件。")
