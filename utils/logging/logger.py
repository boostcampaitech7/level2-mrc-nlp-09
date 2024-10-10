import logging

# 로깅 설정
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

# 로거 설정 예시
logger = setup_logger('my_logger', 'logs/training.log', logging.DEBUG)

# 로그 기록
logger.info('정보 메시지')
logger.error('에러 메시지')
