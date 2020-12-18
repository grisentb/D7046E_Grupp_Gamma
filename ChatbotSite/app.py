from flask import Flask, render_template, request

import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as nps
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
import requests

app = Flask(__name__)
test = ":)"

def guess_sentence(input):

    # labels = {
    #     0: 'negative',
    #     1: 'somewhat negative',
    #     2: 'neutral',
    #     3: 'somewhat positive',
    #     4: 'positive'
    # }

    #vocab = word_vectorizer.vocabulary_

    # input = np.array([sentence])
    # network = nn.Sequential(
    #     nn.Linear(vocab_size, 20),  # First value =  size, Second value = reduced dimension
    #     nn.ReLU(),
    #     nn.Linear(20, len(labels))
    # )
    #
    # network.load_state_dict(torch.load('model.pt'))
    sentence = "This controller does not change the volume"
    # test = torch.zeros(vocab_size)

    sentence = sentence.lower()
    # sentence = sentence.replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # Remove emails
    # sentence = sentence.replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # Remove IP address
    sentence = sentence.replace('!', '')
    # sentence = sentence.replace('[^\w\s]','')                                                       # Remove special characters
    # sentence = sentence.replace('\d', '', regex=True)

    #input = sentence.split()
    # for word in input:
    #     if word in vocab:
    #         test[vocab[word]] += 1

    word_vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=50000, max_df=1, use_idf=True,
                                       norm='l2')
    # input = word_vectorizer2.fit_transform(test)  # Transform texts to sparse matrix
    # input = input.todense()                             # Convert to dense matrix for Pytorch
    # print(type(input))
    # vocab_size = len(word_vectorizer.vocabulary_)
    # input_tensor = torch.from_numpy(np.array(input)).type(torch.FloatTensor)
    # print(input_tensor)
    # prediction = network(test)
    # guess = torch.argmax(prediction, dim=-1)
    #print("Your sentence is", labels[guess.item()])

    return("Your sentence is: " + input)

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST" and 'answer' in request.form:
        question = request.form["answer"]
        test = guess_sentence((question))
        print(test)
        return render_template('index.html', data=test)

    return render_template('index.html', data="")

if __name__ == '__main__':
    app.run(debug=True)


    # vocab is a class that will give us the index for any given word/token (vocab['hi'] = <some number>)
    # vocab = word_vectorizer.vocabulary_
    #
    # trainloader = DataLoader(train_x_tensor, batch_size=10, shuffle=False)
    # trainloader2 = DataLoader(train_y_tensor, batch_size=10, shuffle=False)



