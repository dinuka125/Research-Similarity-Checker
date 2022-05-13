from sentence_transformers import SentenceTransformer, util
from helper import get_sentence_list
import numpy as np
from flask import Flask, render_template, request

model = SentenceTransformer('stsb-roberta-large')


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        sentence = request.form['title']

        print(sentence)
        corpus = get_sentence_list()

        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    

        sentence_embedding = model.encode(sentence, convert_to_tensor=True)

        top_k=5

        cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]

        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
        print("Sentence:", sentence, "\n")
        result_list = []
        print("Top", top_k, "most similar sentences in corpus:")
        for idx in top_results[1:top_k]:
            # print(corpus[idx], "(Score: %.4f)" % (cos_scores[idx])) 
            result_list.append((corpus[idx], "(Score: %.4f)" % (cos_scores[idx])))
            # result_dict[corpus[idx]] = "(Score: %.4f)" % (cos_scores[idx])
            

        return render_template("output.html",items = result_list)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000, debug=True)    