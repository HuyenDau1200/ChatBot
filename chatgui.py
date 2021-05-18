
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

data = json.loads(open('data.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#làm sạch văn bản truyền vào
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# trả về túi từ dạng mảng: 0 hoặc 1 cho mỗi từ tồn tại trong câu
def bow(sentence, words, show_details=True):
    # mã hóa mẫu
    sentence_words = clean_up_sentence(sentence)
    # túi từ - ma trận N từ, ma trận từ vựng
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # chỉ định 1 nếu từ hiện tại ở vị trí từ vựng
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # lọc ra các dự đoán dưới ngưỡng
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sắp xếp theo độ mạnh của xác suất
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, data_json):
    tag = ints[0]['intent']
    list_of_data = data_json['intents']
    for i in list_of_data:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, data)
    return res


# Tạo GUI với tkinter
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Bot")
base.geometry("600x500")
base.resizable(width=FALSE, height=FALSE)

#Tạo cửa sổ chat
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

# Liên kết thanh cuộn với cửa sổ Trò chuyện
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# tạo nút để gửi tin nhắn
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send)

#Tạo hộp để nhập tin nhắn
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)

# đặt vị trí cho các thành phần trong cửa sổ
scrollbar.place(x=576,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=570)
EntryBox.place(x=128, y=401, height=90, width=465)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
