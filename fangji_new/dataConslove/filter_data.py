import json

# 读取JSON文件
with open('CF_utf-8.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 遍历每个方剂信息
for formula, info in data.items():
    # 删除"portion"部分
    if 'portion' in info:
        del info['portion']

    # 如果"components"中有两个条目，则删除第二个条目
    if 'components' in info and len(info['components']) > 1:
        info['components'] = [info['components'][0]]

# 保存修改后的JSON内容
with open('CF_utf-8_modified.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
