
from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__,template_folder='template',static_url_path='/static')
app.config["TEMPLATES_AUTO_RELOAD"] = True
#ADD PKL FILE......model = pickle.load(open("rf_reg.pkl", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")
@app.route("/predictfn",methods =["GET", "POST"])
def predictfn():
    if request.method == "POST":
        cid = request.form.get('cid')
        cdscore = int(request.form.get('cdscore'))
        country= str(request.form.get('country'))
        Gender= str(request.form.get('Gender'))
        age= int(request.form.get('age'))
        tenure= int(request.form.get('tenure'))
        balance= float(request.form.get('balance'))
        pds= int(request.form.get('pds'))
        CreditCard= int(request.form.get('CreditCard'))
        Active= int(request.form.get('Active'))
        salary= float(request.form.get('salary'))

        with open('model1(all_features).pkl', 'rb') as file:
            data = pickle.load(file)

        rand = data['model']
        scaler = data['minmax']
  
        dicti1 = {'France':0,'Germany':1,'Spain':2}
        dicti2 = {'Female':0,'Male':1}
        fd = [[cdscore,age,tenure,balance,pds,CreditCard,Active,salary]]
        df = pd.DataFrame(fd, columns =['CreditScore','Age','Tenure','Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
        df[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']] = scaler.transform(df[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']])
        df = np.array(df)
        gf = [[0,0,0,0,0,df[0][0],df[0][1],df[0][2],df[0][3],df[0][4],df[0][5],df[0][6],df[0][7]]]
        gf[0][dicti1[country]] = 1
        gf[0][int(dicti2[Gender])+3] = 1
        gf = np.array(gf)
        prediction = rand.predict(gf)
        if prediction[0] == 1:
            return render_template('home.html',predict="The customer with CustomerID "+cid+" will exit")
        else:
            return render_template('home.html',predict="The customer with CustomerID "+cid+" will not exit")



        
        





if __name__ == '__main__':
    app.debug = True
    app.run()