import json
import pandas as pd


def wiki_to_title_and_text(path: str, return_path: str):

    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient='index')
    assert 'title' in df.columns and 'text' in df.columns, 'title 또는 text 컬럼이 없습니다.'

    print(
        f"""
        만든 사람: 희준
        
        기능 요약:
        1. title과 text를 하나의 필드로 결합하여 텍스트 일관성 강화.
        2. 데이터 중복 제거: 총 60,613개에서 56,808개로 축소 (중복 항목 제거).
        
        적용 전:
        Title -> {df.iloc[0]["title"]}
        Text Before -> {df.iloc[0]["text"][:50]}"""
    )
    df['text'] = df['title'] + ' ' + df['text']
    print(
        f'''
        적용 후:
        Title -> {df.iloc[0]["title"]}
        Text After  -> {df.iloc[0]["text"][:50+len(df.iloc[0]["title"])+1]}
        '''
    )
    df = df.drop_duplicates(subset='text')
    data = df.to_dict(orient='index')
    
    with open(return_path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    wiki_to_title_and_text('data/raw/wikipedia_documents.json', 'data/preprocessed/wikipedia_documents_no_duplication.json')