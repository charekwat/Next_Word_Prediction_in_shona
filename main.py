#import streamlit as st
import shutil
import os

import numpy as np
from mailbox import ExternalClashError
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import pickle
import numpy as np
import os
import pickle
import streamlit as st
from PIL import Image
import io

# Importing the Libraries

 # load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r' )
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

 # load document
filename = 'hondo_yehumambo_dset.txt'
doc = load_doc(filename)

import re


# Function to clean text and remove numbers
def clean_text(text):
    # Remove numbers using regular expression
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces and newline characters
    text = ' '.join(text.split())

    return text

# Clean the text
cleaned_text = clean_text(doc)

# Print the cleaned text
#print(cleaned_text[:200])
# saving the tokenizer for predict function.
from keras.preprocessing.text import Tokenizer

# Tokenize and preprocess
tokenizer = Tokenizer()
tokenizer.fit_on_texts([cleaned_text])
# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))
total_words = len(tokenizer.word_index) + 1

from gensim.models import Word2Vec


sentences = [sentence.split() for sentence in cleaned_text.split('.')]
model_gensim = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model_gensim.save("word2vec.model")

from tensorflow.keras.models import load_model
import numpy as np
import pickle

import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


input_sequences = []
for line in doc.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)



input_sequences = np.array(pad_sequences(input_sequences, maxlen=6, padding='pre'))
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = keras.utils.to_categorical(y, num_classes=total_words)



# Load the model and tokenizer

model = load_model('best_model2.h5')
tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))




def main():

    """Object detection App"""

    st.title("Next Shona Word Prediction App")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Next Shona word Prediction APP</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.title(" Get the Predictions")
    seed_text = st.text_input("Enter a sentence of five words in shona : ")

    if seed_text is not None:

        try:
            next_words = 1
            suggested_word = []
            # temp = seed_text
            for _ in range(next_words ):
                # Tokenize and pad the text
                sequence = tokenizer.texts_to_sequences([seed_text])[0]
                sequence = pad_sequences([sequence], maxlen=5, padding='pre')

                # Predict the next word
                predicted_probs = model.predict(sequence, verbose=0)
                predicted = np.argmax(predicted_probs, axis=-1)

                # Convert the predicted word index to a word
                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == predicted:
                        output_word = word
                        break

                # Append the predicted word to the text
                #seed_text += " " + output_word

            #return ' '.join(text.split(' ')[-next_words :])

            seed_text += " " + output_word
            print("Suggested next  word  : ", suggested_word)

           # print(seed_text)
        except Exception as e:
            print("Error occurred: ", e)


    if st.button("Suggested_word"):
        st.success(seed_text)


if __name__ == '__main__':
    main()
