import json
import pandas as pd
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer


def split_wiki_for_model(model_name="nlpai-lab/KoE5", path='/data/preprocessed/wikipedia_documents_processed.json', return_path='/data/preprocessed/wikipedia_documents_splitted.json'):
    def sentence_split(text):
        # 간단한 문장 단위 분리 함수 (구두점 기준으로 문장을 분리)
        sentences = re.split(r'(?<=[.!?])\s+|\n+|\\n+|\xa0+|。', text)
        return sentences

    def process_sentences(df, tokenizer, max_seq_length):
        print(
            """
            *******************************************************************
            DataFrame의 각 row에 있는 문장들을 토큰화하고, 주어진 최대 시퀀스 길이에 맞게 문장을 분할하여 처리하는 함수입니다.

            각 row에 대해:
            - 각 문장을 토큰화합니다.
            - 토큰화된 문장들을 합쳐서 최대 시퀀스 길이를 넘기지 않는 범위에서 처리합니다.
            - 토큰 길이가 max_seq_length를 넘으면, 새로운 row를 생성하고 합쳐진 문장들을 저장한 후 다시 길이를 초기화합니다.
            - 모든 문장을 처리할 때까지 반복합니다.

            반환:
            - 처리된 결과가 담긴 새로운 DataFrame을 반환합니다.
            *******************************************************************
        """
        )
        new_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking wiki documents to fit model input size"):
            sentences = row['context_sentences']
            token_length_sum = 0
            current_text = []
            
            for sentence in sentences:
                token_length = len(tokenizer.tokenize(sentence))
                
                if token_length_sum + token_length > max_seq_length:
                    new_row = row.copy()
                    new_row['text_processed'] = " ".join(current_text) # text_processed 대신 text로 나중에 바꾸기
                    new_rows.append(new_row)
                    
                    token_length_sum = 0
                    current_text = []
                
                token_length_sum += token_length
                current_text.append(sentence)
            
            if current_text:
                new_row = row.copy()
                new_row['text_processed'] = " ".join(current_text)
                new_rows.append(new_row)
        
        return pd.DataFrame(new_rows)

    model = SentenceTransformer(
        model_name_or_path=model_name, 
        )
    tokenizer = model.tokenizer
    max_seq_length = min(model.max_seq_length, model[0].auto_model.config.max_position_embeddings)  # 최대 시퀀스 길이

    data_processed = json.load(open(path))
    wiki_p = pd.DataFrame(data_processed).T
    print(wiki_p.shape)
    wiki_p.head()

    tqdm.pandas()

    wiki_p['context_sentences'] = wiki_p['text'].progress_apply(lambda x: sentence_split(x))
    wiki_p.head()

    # Process the DataFrame
    processed_df = process_sentences(df=wiki_p, tokenizer=tokenizer, max_seq_length=max_seq_length)
    print(processed_df.shape)
    processed_df['text'] = processed_df['text_processed']
    processed_df = processed_df.drop(columns=['context_sentences', 'text_processed'])
    processed_df = processed_df.reset_index(drop=True)
    data = processed_df.to_dict(orient='index')

    with open(return_path, 'w') as f:
        json.dump(data, f)
        
        

def drop_title_wiki(path='data/raw/wikipedia_documents.json', return_path='data/preprocessed/wikipedia_documents_no_dup_title_text.json'):

    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient='index')
    
    assert 'title' in df.columns and 'text' in df.columns, 'title 또는 text 컬럼이 없습니다.'

    print(
        f"""
        *******************************************************************
        만든 사람: 희준
        
        기능 요약:
        1. title과 text를 하나의 필드로 결합하여 텍스트 일관성 강화.
        2. 데이터 중복 제거: 총 60,613개에서 56,808개로 축소 (중복 항목 제거).
        
        적용 전:
        Title -> {df.iloc[0]["title"]}
        Text Before -> {df.iloc[0]["text"][:50]}
        
        문서 저장: {return_path}
        """
    )
    df['text'] = df['title'] + ' ' + df['text']
    print(
        f'''
        적용 후:
        Title -> {df.iloc[0]["title"]}
        Text After  -> {df.iloc[0]["text"][:50+len(df.iloc[0]["title"])+1]}
        *******************************************************************
        '''
    )
    df = df.drop_duplicates(subset='text')
    data = df.to_dict(orient='index')
    
    with open(return_path, 'w') as f:
        json.dump(data, f)



def reduce_wiki_length(path='data/preprocessed/wikipedia_documents_no_dup_title_text.json', return_path='data/preprocessed/wikipedia_documents_processed.json'):

    def find_string_in_text(df, search_string):
        return df[df['text'].str.contains(search_string, na=False, regex=False)].index
    
    print(
        f"""
        *******************************************************************
        만든 사람: 희준
        
        기능 요약:
        문장 단위로 wiki 문서를 나눌 때 '한 문장의 토큰 길이가 모델 input보다 큰 경우' 예외처리
        
        문서 저장: {return_path}
        *******************************************************************
        """
    )
    
    
    with open(path) as f:
        data = json.load(f)
    wiki = pd.DataFrame.from_dict(data, orient='index')
    
    text_replacements = [
        ('실제로 2020년 3월 5일 현재까지 단리된 SARS-CoV-2 균주 57개 중, 2019년 3월 5일 현재 단리 일시와 장소를 알 수 없는 UNKNOWN-LR757996 균주(Strain) , SARS-CoV-', [(';', '.')]),
        ('관직으로는 종친부(宗親府)의 군(君)·종정경(宗正卿), 충훈부', [('·', '. ')]),
        ('2003년 ≪또야 너구리가 기운 바지를 입었어요≫(KBS 프로 [TV 책을 말하다] 좋은 어린이 책) ≪수업을 왜 하지?≫(문화관광부 추천도서) ≪땅콩 선생, 드디어', [('도서)', '도서). ')]),
        ('과학기술과 전쟁(크레펠트), 군대 명령과 복종(니코 케이저), 군대문화 이야기(양희완), 군대와 사회(온만금), 군사사상론(군사학연구회), 군사전략입문(에체베리아), 군인(슈나이더), 군인과 국가(헌팅턴), 극한의 경험(유발 하라리), 근대국가와 전쟁(박상섭),', [('),', ').')]),
        ('조선 개국 후 개국원종공신(開國原從功臣)이 되어 내사사인(內史舍人), 병조의랑(兵曹議郞), 세자우필선(世子右弼善), 사헌부 중승(司憲府 中丞), 판교서(判校書)', [(',', '.')]),
        ('제1편인 설회인유편(設會因由篇)부터 엄정팔방편(嚴淨八方篇), 주향통서편(呪香通序篈', [(',', '.')]),
        ('같은 시기에 사용된 다른 연호로는 동진(東晉)에서 사용한 융안(隆安 : 397년 ~ 401년), 원흥(元興 : 402년 ~ 404년), 대형', [(',', '.')]),
        ('곡룡류인 아노돈토사우루스(Anodontosaurus lambei), 에드몬토니아(Edmontoni', [(',', '.')]),
        ('같은 시기에 사용된 다른 연호로는 동진(東晉)에서 사용한 의희(義熙 : 405년 ~ 418년), 원희(原熙 : 419년 ~ 420년), 후진(後秦)에서 사용', [(',', '.')]),
        ('주화산(금남정맥 분리)-모래재-곰티(익산포항고속국도 통과)-슬치(순천완주고속국도, 국도 제17호선, 전라선 철도 통과)-갈미봉(540m)', [('-', '. ')]),
        ('《일본서기》(日本書紀)에는 덴무 천황 13년인 서기 684년', [(',', '.')]),
        ('이와 더불어  故 고우영 작가의 만화를 원작으로 한  애니메이션 《삼국지》', [(',', '.')]),
        ('재미 한인사회 간에 주미외교위원부 문제를 중심으로 분규가 자못 심함으로 1944년 5월 15일 국무회의에서 정·부주석과 외무,', [('하야', '하야. ')]),
        ('또한 궁예가 지향했던 불교와 석총의 불교가 같은 법상종 계열이면서도 궁예는 아미타불, 관음보살 중심이었던 데 반해서 석총은 미륵보살', [('가만히 보고 있을 수 없었을 것이고,', '가만히 보고 있을 수 없었을 것이고. ')]),
        ('스카웃은 게임에서 정말 빠른 클래스이지만, 데미지를 주고 받는게 힘들다.스카웃은 마름쇠로 장전하고 발사하는 못총과 스턴을 멈추고 적을', [('.', '. ')]),
        ('구단주는 우장룡총경리가 담당했다《연변직업축구의 어버이》', [('.', '. ')]),
        ('제1편인 설회인유편(設會因由篇)부터 엄정팔방편(嚴淨八方篇), 주향통서편(呪香通序篇),', [('가지조욕편 제24, ', '가지조욕편 제24. ')]),
        ('《일본서기》(日本書紀)에는 덴무 천황 13년인 서기 684년 음력 11월 1일에 처음으로 아손을 하사받은 52개의 우지로써', [('、', '. ')])
        ]

    for text, replacements in text_replacements:
        index = find_string_in_text(wiki, text)
        for old, new in replacements:
            wiki.loc[index, 'text'] = wiki.loc[index, 'text'].str.replace(old, new, regex=False)
    
    text = '<정세운(鄭世雲)>'
    index = find_string_in_text(wiki, text)
    wiki.loc[index[0], 'text'] = wiki['text'].iloc[int(index[0])].split('<정세운(鄭世雲)>')[0] + text
    
    data = wiki.to_dict(orient='index')
    with open(return_path, 'w') as f:
        json.dump(data, f)
        

if __name__ == '__main__':
    drop_title_wiki(path='data/raw/wikipedia_documents.json',
                           return_path='data/preprocessed/wikipedia_documents_no_dup_title_text.json'
                           )
    reduce_wiki_length(path='data/preprocessed/wikipedia_documents_no_dup_title_text.json', 
                      return_path='data/preprocessed/wikipedia_documents_processed.json'
                      )
    split_wiki_for_model(model_name="nlpai-lab/KoE5", 
           path='data/preprocessed/wikipedia_documents_processed.json', 
           return_path='data/preprocessed/wikipedia_documents_splitted_for_retrieval.json')