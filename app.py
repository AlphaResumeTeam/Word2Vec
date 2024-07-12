from flask import Flask, request, jsonify
import jieba
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 加载预训练的Word2Vec模型
model_path = 'word2vec_model/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
embeddings_index = {}

try:
    with open(model_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # 跳过第一行
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# 定义一个函数来计算句子的平均词向量
def sentence_vector(sentence, embeddings_index):
    words = jieba.lcut(sentence)
    word_vectors = [embeddings_index[word] for word in words if word in embeddings_index]
    if len(word_vectors) == 0:
        return np.zeros(next(iter(embeddings_index.values())).shape[0])
    vector = np.mean(word_vectors, axis=0)
    return vector

# 计算两个句子之间的相似度
def calculate_similarity(text1, text2, embeddings_index):
    vector1 = sentence_vector(text1, embeddings_index)
    vector2 = sentence_vector(text2, embeddings_index)
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity

# 获取数据库中的岗位信息，这里之后应该改成使用仲文的API来获取当前数据库里的所以岗位字段
database_positions = pd.read_excel('high_priority_jobs.xlsx')
database_positions = database_positions['高优先级岗位'].tolist()

@app.route('/similarity', methods=['POST'])
def get_similarity():
    # 获取请求中的 JSON 数据，这里需要改成从前端传过来的用户输入的应聘岗位字段
    data = request.get_json()
    user_input = data.get('user_input', '')

    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    # 计算用户输入与数据库中每个岗位的相似度
    similarities = [(position, calculate_similarity(user_input, position, embeddings_index)) for position in database_positions]

    # 找到相似度最高的岗位
    most_similar_position, highest_similarity = max(similarities, key=lambda x: x[1])

    # 设置阈值
    threshold = 0.8
    marched = highest_similarity > threshold

    # 返回处理后的数据
    return jsonify({
        'most_similar_position': most_similar_position,
        'highest_similarity': highest_similarity,
        'matched': marched
    })

# 这里需要改成我们的服务器地址
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
