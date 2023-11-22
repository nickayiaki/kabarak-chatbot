from flask import Flask, render_template, request,url_for
import random
import json
from keras.models import load_model
import webbrowser
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('popular')
lemmatizer = WordNetLemmatizer()
model = load_model('model.kabarak')
intents = json.loads(open('intents.json', "r+", encoding="utf-8").read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for wordb 
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    if p.shape[0] < 117:
        pad_width = 117 - p.shape[0]
        p = np.pad(p, (0, pad_width), 'constant')
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:list
    return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
        
    return result
def chatbot_response(msg):
    # Handle empty or non-meaningful input
    if not msg.strip():
        return "Please enter a meaningful message."

    # Handle generic or random input
    generic_responses = [
        "I'm not sure what you're trying to say.",
        "Could you please provide more context?",
        "I didn't understand that. Can you rephrase your question?",
        "Sorry, I couldn't comprehend your message.",
    ]

    ints = predict_class(msg, model)

    if ints:
        response = getResponse(ints, intents)
        if response:
            return response

    # Return a random generic response if no matching intent is found
    return random.choice(generic_responses)



def chatbot_response(msg):
    ints = predict_class(msg, model)

    if ints:
        response = getResponse(ints, intents)
        if response:
            return response

    # Handle the case when 'ints' is empty or no matching intent is found
    return "I'm sorry, I don't understand that."




app = Flask(__name__, template_folder='template')
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    open_browser()
    app.run()