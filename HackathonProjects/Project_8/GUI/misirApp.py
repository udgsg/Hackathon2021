from tkinter import *
import time
import pickle
from training import training_model
from testing import testing_model

def click_training():
    output_status.insert(INSERT, "Training model...")
    #C:/Users/amahbub/Documents/personal/misinformation_hackathon/dataset_3/train.csv
    data_path = textEntry.get()
    global file_name
    global model 
    if str(var.get())=="1":
        acc,model= training_model(data_path)
        print(acc)
        file_name="model_1.sav"
        #pickle.dump(model, open(file_name, 'wb'))
    elif str(var.get())=="2":
        print("training_2.py")
        acc,model= training_model(data_path)
        print(acc)
        file_name="model_2.sav"
        #pickle.dump(model, open(file_name, 'wb'))
    
    try:
        definition = mydict[data_path]
    except:
        definition= "sorry, no words found"
    
    time.sleep(2)
    output_status.delete(0.0, END) 
    output_status.insert(END, "Training completed! \n Model Accuracy: "+str(acc*100)+"%"+"\n")
    print(model.summary())

def click_prediction():
    global file_name
    global model
    test_str = textEntry_pred.get()
    #process test_str
    X_test = testing_model(test_str)
    
    #loaded_model = pickle.load(open(file_name, 'rb'))
    pred = model.predict(X_test)
    #output_pred = loaded_model.score(X_test, Y_test)
    if pred == 1: dec = "Fake"
    elif pred == 0: dec = "Real"
    output_pred.inter(END, "The article is: "+ dec +".")

##############################################################################
file_name = ""
w1=Tk()
w1.title("MisInformation Remover (MISIR)")
# Width, height in pixels
f1=Frame(w1, height=500, width=1000)

#f1.pack()
#background photo
photo1 = PhotoImage(file="./resources/1.PNG")
Label(w1, image = photo1, bg="white") .grid(row=0,column=0, sticky=W) 
#####################################################################
#getting data address
Label(w1, text="Data path", font="none 12 bold") .grid(row=1, column=0, sticky=W)
#text entry box
textEntry = Entry(w1, width=60, bg="white")
textEntry.grid(row=2, column=0, sticky=W)

#radio button: 1. title+Author  or text
var = IntVar()
R1 = Radiobutton(w1, text="Titel+Author", variable=var, value=1).grid(row=3, column=0, sticky=W)
R2 = Radiobutton(w1, text="Article body", variable=var, value=2).grid(row=4, column=0, sticky=W)



#submit button for training
Button(w1, text="Submit", width=6, command=click_training) .grid(row=5, column=0, sticky=W)
#training status
Label(w1, text="\nTraining status", font="none 12 bold") .grid(row=6, column=0, sticky=W)
output_status = Text(w1, width=40, height=3, wrap=WORD, background="white")
output_status.grid(row=7, column=0, columnspan=2, sticky=W)
#output_status.insert(INSERT, "Training model...")


#################################################################
#prediction
divider_str = "========================================================================="
Label(w1, text=divider_str, font="none 12 bold") .grid(row=8, column=0, sticky=W)
#Label(w1, text="\n\n Enter Author+Titel or Text", font="none 12 bold") .grid(row=9, column=0, sticky=W)
#getting output text
Label(w1, text="\n\n Enter Author+Titel or Text:", font="none 12 bold") .grid(row=10, column=0, sticky=W)
#text entry box
textEntry_pred = Entry(w1, width=60, bg="white")
textEntry_pred.grid(row=11, column=0, sticky=W)
#submit button for prediction
Button(w1, text="Predict", width=6, command=click_prediction) .grid(row=12, column=0, sticky=W)

output_pred = Text(w1, width=75, height=2, wrap=WORD, background="white")
output_pred.grid(row=13, column=0, columnspan=2, sticky=W)


#outer buffer
Label(w1, text="\n\n", font="none 12 bold") .grid(row=14, column=0, sticky=W)

mydict = {'algo':"step by step", 'bug': "debugging phase"}
w1.mainloop()
    