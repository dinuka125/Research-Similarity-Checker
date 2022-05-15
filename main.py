from sentence_transformers import SentenceTransformer, util
from helper import get_sentence_list
import numpy as np
from flask import Flask, render_template, request

# model = SentenceTransformer('stsb-roberta-large')
model = SentenceTransformer('all-MiniLM-L6-v2')


app = Flask(__name__)
corpus = get_sentence_list()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        sentence = request.form['title']

        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    

        sentence_embedding = model.encode(sentence, convert_to_tensor=True)

        top_k=10

        cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]

        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
        print("Sentence:", sentence, "\n")
        result_list = []
        print("Top", top_k, "most similar sentences in corpus:")
        for idx in top_results[1:top_k]:
            # print(corpus[idx], "(Score: %.4f)" % (cos_scores[idx])) 
            result_list.append((corpus[idx], "Score: %.4f" % ((cos_scores[idx])*100)))
            # result_dict[corpus[idx]] = "(Score: %.4f)" % (cos_scores[idx])
            

        return render_template("output.html",items = result_list)

 