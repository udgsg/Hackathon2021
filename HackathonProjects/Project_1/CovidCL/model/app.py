from flask import Flask,render_template,url_for,request
import pickle
import hackaton_functions as hf
import joblib
#from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/Confirm',methods=['POST'])
def Confirm():
	""" Generates prediction for a user input text
	"""
	covid_fake_model = open('final_model_2.pkl','rb')
	clf,column_names = pickle.load(covid_fake_model)
	
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		row=hf.get_row(data,column_names)
	
		prediction=hf.get_prediction(clf,row)
	return render_template('result.html',prediction = prediction)


if __name__ == '__main__':
	app.run(debug=True)