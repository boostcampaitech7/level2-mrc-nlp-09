from konlpy.tag import Kkma

class QueryPreprocessor:
    """
    Query 전처리 모듈화 클래스
    전처리 방법을 모듈화하고, 전처리를 사용할지 여부를 제어할 수 있습니다.
    """
    def __init__(self):
        self.kkma = Kkma()

    def preprocess(self, query: str) -> str:
        """
        Query에 대한 전처리 작업을 수행하는 함수.
        형태소 분석을 통해 의미 있는 태그(명사, 동사, 형용사 등)를 추출하고, 
        원래 query 앞에 공백으로 구분하여 추가합니다.
        """
        # 의미 있는 품사 태그 목록
        meaningful_pos_tags = ['NNG', 'NNP', 'VV', 'VA']
        
        # 형태소 분석 수행
        tagged_tokens = self.kkma.pos(query)
        
        # 의미 있는 토큰 필터링
        meaningful_tokens = [token for token, pos in tagged_tokens if pos in meaningful_pos_tags]
        
        # 의미 있는 토큰을 공백으로 구분하여 하나의 문자열로 연결
        filtered_text = ' '.join(meaningful_tokens)
        
        # 기존 query에 필터링된 결과를 앞에 추가
        processed_query = f"{filtered_text} {query}"
        
        return processed_query
    
