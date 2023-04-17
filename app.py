from flask import Flask, jsonify, request, render_template
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
from spacy.lang.en import English
import json

app = Flask(__name__)

class PreProcessing:
    def split_chars(self,text):
        return " ".join(text)

    def spliting_text(self,text):
        nlp = English()
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        return [str(t)for t in doc.sents]

    def formating_the_data(self,text):
        Total_Number_Of_Lines = len(text)
        whole_Data = []
        for i,lines in enumerate(text):
            d = {}
            d["Text"] = lines
            d["Line_Number"] = i
            d["Total_Number_Of_Lines"] = Total_Number_Of_Lines - 1
            whole_Data.append(d)
        return whole_Data
    
    def labeling_the_data(self,data):
        test_abstract_lines_number = [line["Line_Number"]for line in data]
        test_abstract_lines_number_one_hot = tf.one_hot(test_abstract_lines_number,depth=15)
        test_abstract_total_lines = [line["Total_Number_Of_Lines"]for line in data]
        test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines,depth=20)
        return test_abstract_lines_number_one_hot,test_abstract_total_lines_one_hot
    
    def creating_chars(self,text):
        return [self.split_chars(i) for i in text]
    
    def classes(self):
        return ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']

print(">>Loading the Model<<")    
Loaded_Model = tf.keras.models.load_model("QuickieLit",custom_objects={"Keras_Layer":hub.KerasLayer})
print(">>Finished<<")

@app.route("/")
def index():
    return render_template("index.html")


def predict(text):

    PreProcessing_Amelia = PreProcessing()

    Text = PreProcessing_Amelia.spliting_text(text)
    Text_Chars = PreProcessing_Amelia.creating_chars(Text)
    Text_Dictonary = PreProcessing_Amelia.formating_the_data(Text)
    text_lines_number,text_total_lines = PreProcessing_Amelia.labeling_the_data(Text_Dictonary)
    classes = PreProcessing_Amelia.classes()
    
    test_pred = Loaded_Model.predict(x=(tf.constant(Text),tf.constant(Text),tf.constant(Text_Chars),
                                        text_lines_number,text_total_lines))
    
    test_pred_le = tf.argmax(test_pred,axis=1)
    test_predicted_class = [classes[i]for i in test_pred_le]
    pred_probs = tf.reduce_max(test_pred,axis=1)
    results = []
    for i, text in enumerate(Text):
        result = {
            "predicted_class": test_predicted_class[i],
            "probability": f"{pred_probs[i]*100:.2f}%",
            "text": text
        }
        results.append(result)
    return results

@app.route("/predict", methods=["POST"])
def make_prediction():
    text = request.json["text"]
    results = predict(text)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
