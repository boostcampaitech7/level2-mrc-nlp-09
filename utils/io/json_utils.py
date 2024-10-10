import json

# JSON 파일 읽기
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# JSON 파일 쓰기
def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
