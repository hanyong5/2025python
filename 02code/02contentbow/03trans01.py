from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("bert-base-uncased")
# model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# sentences = [
#     "부산 여행 가고 싶다",
#     "부산 바다 보고 싶다",
#     "주식 시장이 어렵다"
# ]
sentences = [
    "I want to travel to Busan",
    "I want to see the sea in Busan",
    "The stock market is difficult"
]

embeddings = model.encode(sentences)
similarity = util.cos_sim(embeddings[0],embeddings[1])
similarity1 = util.cos_sim(embeddings[0],embeddings[2])

print("문장 1",sentences[0] )
print("문장 2",sentences[1] )
print("유사도 점수",float(similarity) )

print("문장 1",sentences[0] )
print("문장 2",sentences[2] )
print("유사도 점수",float(similarity1) )