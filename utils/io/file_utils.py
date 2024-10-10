# 텍스트 파일 읽기
def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

# 텍스트 파일 쓰기
def write_file(data, file_path):
    with open(file_path, 'w') as f:
        f.write(data)
