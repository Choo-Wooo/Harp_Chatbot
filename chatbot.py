# curl ipinfo.io/ip 을 통해 현재 ip 찾기


import flask
from flask_cors import CORS
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_restx import Resource, Api, reqparse
from functools import cache
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm

app = Flask(__name__)
CORS(app)
api = Api(app)
app.chat=[]
app.config['DEBUG'] = True

@cache
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@cache
def get_dataset():
    df = pd.read_csv('wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()




def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def return_similar_answer(input) :
    embedding = model.encode(input)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    return df.loc[df['distance'].idxmax()]['챗봇']

print(return_similar_answer('너무 우울해'))


@app.route('/chatbot')
def chat_user():
    chat = request.args.get('chat')
    jsonResult=[]
    chat_answer = return_similar_answer(chat)
    jsonResult.append({'answer':chat_answer})
    jsonFile=json.dumps(jsonResult, indent=4, sort_keys=True, ensure_ascii=False)
    return jsonFile



if __name__ == "__main__":
    app.run(host="0.0.0.0", port="80", debug=True)
"""
@app.route('/chatbot_user')




st.header('Harp Chatbot')


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('당신: ', '')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

for i in reversed(range(len(st.session_state['past']))):
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
"""
