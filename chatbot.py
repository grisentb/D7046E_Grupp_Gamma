import nltk
nltk.download('punkt')
nltk.download('stopwords')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # Remove emails
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # Remove IP address
    data['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')                                                       # Remove special characters
    data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)                                                   # Remove numbers
    for index, row in data.iterrows():
        print(f'\rEpoch {index}', end='')
        word_tokens = word_tokenize(row['Sentence'])
        filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
        df_ = df_.append({
            "index": row['index'],
            "Class": row['Class'],
            "Sentence": " ".join(filtered_sent[0:])
        }, ignore_index=True)
    return data

# If this is the primary file that is executed (ie not an import of another file)
# if __name__ == "__main__":
#     # Get data, pre-process and split
#     data = pd.read_csv("sample_data/amazon_cells_labelled.txt", delimiter='\t', header=None)
#     data.columns = ['Sentence', 'Class']
#     data['index'] = data.index                                          # Add new column index
#     columns = ['index', 'Class', 'Sentence']
#     data = preprocess_pandas(data, columns)                             # Pre-process
#     training_data, validation_data, training_labels, validation_labels = train_test_split( # split the data into training, validation, and test splits
#         data['Sentence'].values.astype('U'),
#         data['Class'].values.astype('int32'),
#         test_size=0.10,
#         random_state=0,
#         shuffle=True
#     )
#     unvectorized_training_data = training_data
#     # Vectorize data using TFIDF and transform for PyTorch for scalability
#     word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000, max_df=0.5, use_idf=True, norm='l2')
#     training_data = word_vectorizer.fit_transform(training_data)        # Transform texts to sparse matrix
#     training_data = training_data.todense()                             # Convert to dense matrix for Pytorch
#     print(type(training_data))
#     vocab_size = len(word_vectorizer.vocabulary_)
#     validation_data = word_vectorizer.transform(validation_data)
#     validation_data = validation_data.todense()
#     train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.FloatTensor)
#     train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
#     validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
#     validation_y_tensor = torch.from_numpy(np.array(validation_labels)).long()

if __name__ == "__main__":
    # Get data, pre-process and split
    data = pd.read_csv("sample_data/train2.tsv", delimiter='\t', header=None)
    data.columns = ['index', 'SentenceId',	'Sentence', 'Class']
    # data['index'] = data.index                                          # Add new column index
    columns = ['index', 'SentenceId',	'Sentence', 'Class']
    data = preprocess_pandas(data, columns)                             # Pre-process
    training_data, validation_data, training_labels, validation_labels = train_test_split( # split the data into training, validation, and test splits
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True
    )
    unvectorized_training_data = training_data
    # Vectorize data using TFIDF and transform for PyTorch for scalability
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000, max_df=0.5, use_idf=True, norm='l2')
    training_data = word_vectorizer.fit_transform(training_data)        # Transform texts to sparse matrix
    training_data = training_data.todense()                             # Convert to dense matrix for Pytorch
    print(type(training_data))
    vocab_size = len(word_vectorizer.vocabulary_)
    validation_data = word_vectorizer.transform(validation_data)
    validation_data = validation_data.todense()
    train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.FloatTensor)
    train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
    validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
    validation_y_tensor = torch.from_numpy(np.array(validation_labels)).long()

labels = {
    0 : 'Negative',
    1 : 'Positive',
}

# vocab is a class that will give us the index for any given word/token (vocab['hi'] = <some number>)
vocab = word_vectorizer.vocabulary_

print(
    f'Chatbot training data contains {len(training_data)} labelled text snippets '
    f'and has a vocabulary size of {len(vocab)}\n'
    f'On index 0, Chatbot training data contains:\n{training_data[0]}\n'
    f'which has the label {train_y_tensor[0]} that translates into "{labels[train_y_tensor[0].item()]}" '
    f'and the text snippet itself translates into:\n{unvectorized_training_data[0]}'
)

labels = {
    0 : 'negative',
    1 : 'somewhat negative',
    2 : 'neutral',
    3 : 'somewhat positive',
    4 : 'positive'
}

# vocab is a class that will give us the index for any given word/token (vocab['hi'] = <some number>)
vocab = word_vectorizer.vocabulary_

print(
    f'Chatbot training data contains {len(training_data)} labelled text snippets '
    f'and has a vocabulary size of {len(vocab)}\n'
    f'On index 0, Chatbot training data contains:\n{training_data[0]}\n'
    f'which has the label {train_y_tensor[0]} that translates into "{labels[train_y_tensor[0].item()]}" '
    f'and the text snippet itself translates into:\n{unvectorized_training_data[0]}'
)

trainloader = DataLoader(train_x_tensor, batch_size=10, shuffle=False)
trainloader2 = DataLoader(train_y_tensor, batch_size=10, shuffle=False)

network = nn.Sequential(
  nn.Linear(vocab_size, 20), # First value =  size, Second value = reduced dimension
  nn.ReLU(),
  nn.Linear(20, len(labels))
    )

optimizer = torch.optim.Adam(network.parameters())
loss_function = nn.CrossEntropyLoss()
epochs = 5 # The dataset is large so one epoch should do for our purpose (and anything more would take forever)

for epoch in range(epochs):
    # For each batch of data (since the dataset is too large to run all data through the network at once)
    for batch_nr, (data, label) in enumerate(zip(trainloader, trainloader2)):
        
        prediction = network(data)

        #predictcion = torch.argmax(prediction[batch_nr], dim=-1)
        
        # Calculate the loss of the prediction by comparing to the expected output
        loss = loss_function(prediction, label)
        
        # Backpropagate the loss through the network to find the gradients of all parameters
        loss.backward()
        
        # Update the parameters along their gradients
        optimizer.step()
        
        # Clear stored gradient values
        optimizer.zero_grad()
        
        #Print the epoch, batch, and loss
        print(f'\rEpoch {epoch+1} [{batch_nr+1}/{len(trainloader)}] - Loss: {loss}', end='')

#Validation Step
testloader = DataLoader(validation_x_tensor, batch_size=1, shuffle=False)
testloader2 = DataLoader(validation_y_tensor, batch_size=1, shuffle=False)

with torch.no_grad():
    correct = 0
    total = 0
    # For each batch of testing data (since the dataset is too large to run all data through the network at once)
    for batch_nr, (data, label) in enumerate(zip(testloader, testloader2)):
        prediction = network(data)

        total += 1

        guess = torch.argmax(prediction, dim=-1)

        if guess.item() == label.item():
            correct += 1
    
    print(f'The accuracy of the network is {str(100*correct/total)[:4]}%.')

# Let's print a sentence and predict it's category
sentence_index = 0

prediction = network(train_x_tensor[0])

guess = torch.argmax(prediction, dim=-1)

print(
    f'The network predicted that \n"{unvectorized_training_data[0]}"\n should be in the category {labels[guess.item()]}'
)

# Take input from user and classify
sentence = input()

#input = np.array([sentence])
sentence = "This controller does not change the volume"
test = torch.zeros(vocab_size)

sentence = sentence.lower()
#sentence = sentence.replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # Remove emails
#sentence = sentence.replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # Remove IP address
sentence = sentence.replace('!','')   
#sentence = sentence.replace('[^\w\s]','')                                                       # Remove special characters
#sentence = sentence.replace('\d', '', regex=True) 

input = sentence.split()
for word in input :
  if word in vocab:
    test[vocab[word]] +=1

word_vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000, max_df=1, use_idf=True, norm='l2')
#input = word_vectorizer2.fit_transform(test)  # Transform texts to sparse matrix
#input = input.todense()                             # Convert to dense matrix for Pytorch
#print(type(input))
# vocab_size = len(word_vectorizer.vocabulary_)
#input_tensor = torch.from_numpy(np.array(input)).type(torch.FloatTensor)
#print(input_tensor)
prediction = network(test)
guess = torch.argmax(prediction, dim=-1)

print("Your sentence is", labels[guess.item()])
