import pickle
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model('predicter.keras')

with open('tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)

def predict_next_word(model,tokenizer,text,max_seq_len):
    token_list=tokenizer.texts_to_sequences([text])[0]

    if len(token_list)>=max_seq_len:
        token_lis=token_list[-(max_seq_len-1):]

    token_list=pad_sequences([token_list],maxlen=max_seq_len-1,padding='pre')

    predicted=model.predict(token_list,verbose=0)

    predicted_word_index=np.argmax(predicted,axis=1)

    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None
st.title('Next Word Predictor')

input_text=st.text_input("enter the sequence of words")
if(st.button('Predict Next Word')):
    max_seq_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_seq_len)
    st.write(f'Next word can be:{next_word}')