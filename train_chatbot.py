import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json',encoding='utf-8').read()
data = json.loads(data_file)


for intent in data['intents']:
    for pattern in intent['patterns']:

        # lấy từng từ và mã hóa nó
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # thêm tài liệu
        documents.append((w, intent['tag']))

        # thêm các lớp tag vào danh sách classes hiện có trong file classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# khởi tạo dữ liệu đào tạo
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # khởi tạo túi từ
    bag = []
    # danh sách các từ được mã hóa cho mẫu
    pattern_words = doc[0]
    # lemmatize từng từ - tạo từ cơ bản, cố gắng biểu diễn các từ liên quan
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # tạo mảng túi từ của chúng ta với 1, nếu tìm thấy kết hợp từ trong mẫu hiện tại
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # đầu ra là '0' cho mỗi thẻ và '1' cho thẻ hiện tại (cho mỗi mẫu)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# xáo trộn các tính năng và biến thành np.array
random.shuffle(training)
training = np.array(training)
# tạo danh sách đào tạo và kiểm tra. X - mẫu, Y - ý định
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Tạo mô hình - 3 lớp. Lớp đầu tiên 128 neurons, lớp thứ hai 64 neurons và lớp đầu ra thứ 3 chứa số lượng neurons
# bằng số ý định để dự đoán ý định đầu ra với softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# Biên dịch mô hình. Đổ dốc ngẫu nhiên với gia tốc Nesterov mang lại kết quả tốt cho mô hình này
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#cài đặt và lưu mô hình
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
