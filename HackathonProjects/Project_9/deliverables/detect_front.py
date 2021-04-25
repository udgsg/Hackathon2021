import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings, time
#for summarization
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
#for token classification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
#for html
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#import category as category
# App config.
DEBUG = True
app = Flask(__name__)
run_with_ngrok(app)   #starts ngrok when the app is run
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'randomthing'


class ReusableForm(Form):
    name = TextField('Name:')#, validators=[validators.required()])

@app.route("/", methods=['GET', 'POST'])
def hello():
        form = ReusableForm(request.form)
        #name = TextField('Name:', validators=[validators.required()])
        print(form.errors)
        if request.method == 'POST':
            name=request.form['text']
            print(name)
            if form.validate():
                # Save the comment here.
                flash(name)
                if request.form['action']=="Summarization":
                  flash("Summary:")
                  value = get_summary(name)
                  flash(value)
                elif request.form['action'] =="Tokenization":
                  flash("Token(s):")
                  value = get_keys(name)
                  flash(value)
            else:
                flash('Error.')
 
        return render_template('front.html', form=form)


def get_summary(text):
        try:
            model_name = 'google/pegasus-xsum'
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
            src_text=[""""""+text+""""""]
            batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)
            translated = model.generate(**batch)
            target = tokenizer.batch_decode(translated, skip_special_tokens=True)
        except :
            print("API Error occured")
            return (-100)
        return target[0]

def get_keys(text):
        try:
            src_text=[""""""+text+""""""]
            tokenizer = AutoTokenizer.from_pretrained("elastic/distilbert-base-cased-finetuned-conll03-english")
            model = AutoModelForTokenClassification.from_pretrained("elastic/distilbert-base-cased-finetuned-conll03-english")
            nlp = pipeline("ner", model=model, tokenizer=tokenizer)
            ner_results = nlp(src_text)
            print(type(ner_results))
        except :
            print("Error occured")
            return (-100)
        return ner_results

if __name__ == '__main__':
    app.run()

    
