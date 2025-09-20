import json
import random

# 假设JSON文件名为'CF_utf-8_modified.json'
json_file = 'CF_utf-8_modified.json'

# 加载JSON数据
with open(json_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 定义一些随机值的选项
dosage_forms = ["丸剂", "汤剂", "祛痰剂", "栓剂", "膏剂"]
applicable_people = ["成人", "儿童", "老人", "孕妇", "所有人群"]
contraindications = ["对某些成分过敏者禁用", "孕妇禁用", "儿童慎用", "无禁忌"]
administration_methods = ["口服", "外用", "注射", "涂抹"]
precautions = ["使用前请咨询医师", "避免与其他药物同时使用", "定期复查", "不要超量服用"]

# 初始化SQL脚本列表
sport_info_sql = []
detail_sql = []

# 遍历JSON数据生成SQL插入语句
id_counter = 1
for name, info in data.items():
    # 随机选择剂型、适用人群、禁忌、服用方式和注意事项
    dosage_form = random.choice(dosage_forms)
    applicable_person = random.choice(applicable_people)
    contraindication = random.choice(contraindications)
    administration_method = random.choice(administration_methods)
    precaution = random.choice(precautions)

    # 生成功能主治字符串
    function_treatment = '，'.join(info['disease'])

    # 生成成分字符串
    components = info['components'][0]
    components_str = '，'.join([f"{k}{v}克" for k, v in components.items()])

    # 生成SQL插入语句
    sport_info_sql.append(f"INSERT INTO 'sport_info' VALUES({id_counter}, '{name}', '{dosage_form}', '{applicable_person}', '{function_treatment}', '{contraindication}');")
    detail_sql.append(f"INSERT INTO 'detail' VALUES({id_counter}, '{name}', '{components_str}', '{administration_method}', '{precaution}');")

    id_counter += 1

# 输出SQL脚本
with open('insert_sport_info.sql', 'w', encoding='utf-8') as file:
    file.write('\n'.join(sport_info_sql))

with open('insert_detail.sql', 'w', encoding='utf-8') as file:
    file.write('\n'.join(detail_sql))

print("SQL脚本已生成并保存到文件中。")
