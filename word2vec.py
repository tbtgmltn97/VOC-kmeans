import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

train_df = pd.read_csv("/content/train_df.csv",encoding = 'cp949', engine='c')
train_data = train_df[['발화문','상담번호']]

# 불용어 제거하고 형태소분석으로 명사만 뽑음
# 불용어 정의
stopwords = ['을', 'ㅋㅋ', '부터', '까지' '적', '의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()

tokenized_data = []
for sentence in tqdm(train_data):
    tokenized_sentence = okt.nouns(sentence) # 토큰화 #okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)

from gensim.models import Word2Vec

model = Word2Vec(sentences = tokenized_data, size = 50, #몇 개의 숫자로 나타낼지를 지정
                 window = 5, min_count = 5, workers = 4, sg = 0)


from sklearn.cluster import KMeans
model1 = KMeans(n_clusters=3, random_state=0, algorithm='auto')
model1.fit(model.wv.vectors)

pred=model1.predict(model.wv.vectors)

