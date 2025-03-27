import json

def convert_format(input_json):
    conversations = input_json['conversations']
    new_conversations = []
    for conv in conversations:
        from_role = "human" if conv['role'] == 'user' else 'gpt'
        new_conversations.append({
            "from": from_role,
            "value": conv['content']
        })

    return {
        "id": input_json['image'][:-4],  # 根据需要自行设定id
        "image": input_json['image'],  # 根据需要自行设定image
        "conversations": new_conversations
    }

# 批量转换
def batch_convert(input_list):
    return [convert_format(item) for item in input_list]
# 示例用法

data=[]
with open('/mnt/data/shenxiaochen_data/multimodel/sft_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

converted_data = batch_convert(data)

with open('sft_data_output.json', 'w', encoding='utf-8') as file:
    json.dump(converted_data, file, ensure_ascii=False, indent=4)
