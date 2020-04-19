# %% [code]
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
import csv
import re
import os
from IPython.display import FileLink
import pandas as pd


# %% [code]
stopwords =  [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

# %% [code]
num_words = 30000
embedding_size = 128
epochs = 3
oov_tok = '<OOV>' 
padding = 'post'
max_len = 40
trunc_type = 'post'

# %% [code]
train_sentences = []
train_labels = []
validation_sentences = []
validation_labels = []
sentences = []
labels = []
with open('../input/finding-chandler-sarcasm/train.csv') as trainFile:
    reader = csv.reader(trainFile,delimiter=',')
    next(reader)
    for row in reader:
        sentence = row[2].lower()
        sentence = ' '.join(re.sub("(@[A-Za-z0-9]+)",'',sentence).split())
        sentence = ' '.join(re.sub('(#[A-Za-z0-9]+)','',sentence).split())
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)
        labels.append(row[1])
limit = (len(sentences)*90)//100
train_sentences = sentences[:limit]
validation_sentences = sentences[limit:]
train_labels = labels[:limit]
validation_labels = labels[limit:]
tokenizer = Tokenizer(num_words = num_words, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

# %% [code]
train_sequences = tokenizer.texts_to_sequences(train_sentences)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)


# %% [code]
train_padded = pad_sequences(train_sequences, maxlen=max_len,padding=padding, truncating = trunc_type)
validation_padded = pad_sequences(validation_sequences, maxlen=max_len, padding=padding, truncating = trunc_type)


# %% [code]
model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, embedding_size, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,kernel_regularizer=regularizers.l2(0.001))),
    tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# %% [code]
model2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, embedding_size, input_length=max_len),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(54, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# %% [code]
model3 = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, embedding_size, input_length=max_len),
    tf.keras.layers.Conv1D(256,10,activation = 'relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(54, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% [code]
model4 = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, embedding_size, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16,kernel_regularizer=regularizers.l2(0.001))),
    tf.keras.layers.Dense(400, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% [code]
model5 = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, embedding_size, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32,kernel_regularizer=regularizers.l2(0.001))),
    tf.keras.layers.Dense(200, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model5 .compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% [code]
hitstory1 = model1.fit(train_padded, np.array(train_labels), epochs=epochs, validation_data=(validation_padded, np.array(validation_labels)))

# %% [code]
hitstory2 = model2.fit(train_padded, np.array(train_labels), epochs=epochs, validation_data=(validation_padded, np.array(validation_labels)))

# %% [code]
hitstory3 = model3.fit(train_padded, np.array(train_labels), epochs=epochs, validation_data=(validation_padded, np.array(validation_labels)))

# %% [markdown]
# 

# %% [code]
hitstory4 = model4.fit(train_padded, np.array(train_labels), epochs=epochs, validation_data=(validation_padded, np.array(validation_labels)))

# %% [code]
hitstory5 = model5.fit(train_padded, np.array(train_labels), epochs=epochs, validation_data=(validation_padded, np.array(validation_labels)))

# %% [code]
model_array = [model1,model2,model3,model4,model5]
model_predictions_on_train_data = []
model_predictions_on_validation_data = []
final_model_train_labels = train_labels
final_model_validation_labels = validation_labels
for model in model_array:
    model_predictions_on_train_data.append([item[0] for item in  list(model.predict(train_padded))])
    model_predictions_on_validation_data.append([item[0] for item in list(model.predict(validation_padded))])
final_model_train_data = []
final_model_validation_data = []
for i in range(len(model_predictions_on_train_data[0])):
    lst = []
    for j in range(len(model_array)):
        lst.append(model_predictions_on_train_data[j][i])
    final_model_train_data.append(lst)
for i in range(len(model_predictions_on_validation_data[0])):
    lst = []
    for j in range(len(model_array)):
        lst.append(model_predictions_on_validation_data[j][i])
    final_model_validation_data.append(lst)

# %% [code]
final_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% [code]
hitstory_final_model = final_model.fit(np.array(final_model_train_data), np.array(final_model_train_labels),epochs=5, validation_data=(np.array(final_model_validation_data), np.array(final_model_validation_labels)))

# %% [code]
final_model_train_data+final_model_validation_data

# %% [code]
total_train_data = final_model_train_data+final_model_validation_data
np.array(total_train_data).shape

# %% [code]
pred = final_model.predict(np.array(total_train_data))
indexes = set()
for i in range(len(pred)):
    if pred[i][0]>=0.3 or pred[i][0]<=0.7:
        indexes.add(i)

total_train_labels = final_model_train_labels+final_model_validation_labels
train_data_correction_model = []
train_labels_correction_model = []
for i in range(len(total_train_data)):
    if i in indexes:
        train_data_correction_model.append(total_train_data[i])
        train_labels_correction_model.append(total_train_labels[i])


# %% [code]
correction_model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(24, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
correction_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% [code]
hitstory_correction_model = correction_model.fit(np.array(train_data_correction_model), np.array(train_labels_correction_model),epochs=5)

# %% [code]


# %% [code]
test_sentences = []
with open('../input/finding-chandler-sarcasm/test.csv') as testFile:
    reader = csv.reader(testFile,delimiter=',')
    next(reader)
    for row in reader:
        sentence = row[1].lower()
        sentence = ' '.join(re.sub("(@[A-Za-z0-9]+)",'',sentence).split())
        sentence = ' '.join(re.sub('(#[A-Za-z0-9]+)','',sentence).split())
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
        test_sentences.append(sentence)
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_len,padding=padding, truncating = trunc_type)

# %% [code]
model_predictions_on_test_data = []
for model in model_array:
    model_predictions_on_test_data.append([item[0] for item in  list(model.predict(test_padded))])
final_model_test_data = []
for i in range(len(model_predictions_on_test_data[0])):
    lst = []
    for j in range(len(model_array)):
        lst.append(model_predictions_on_test_data[j][i])
    final_model_test_data.append(lst)

# %% [code]
output = final_model.predict(np.array(final_model_test_data))

# %% [code]

indexes = set()
for i in range(len(output)):
    if output[i][0]>=0.3 or output[i][0]<=0.7:
        indexes.add(i)
test_data_correction_model = []
for i in range(len(final_model_test_data)):
    if i in indexes:
        test_data_correction_model.append(final_model_test_data[i])


# %% [code]
finalResult = correction_model.predict(np.array(test_data_correction_model))

# %% [code]
import pandas as pd
Ids = []
result = []
j = 0
for i in range(len(output)):
    res = 0
    if i in indexes:
        output[i][0] = finalResult[j][0]
        j = j + 1
    if output[i][0] >= 0.5:
        res = 1
    Ids.append(i)
    result.append(res)
    i = i+1
print(len(result))
csvData = pd.DataFrame({
    'Id': Ids,
    'label': result
})
csvData.to_csv('result15.csv', index = False)

# %% [code]
FileLink(r'result15.csv')


# %% [code]

