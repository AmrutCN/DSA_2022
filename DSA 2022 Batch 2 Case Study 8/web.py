from flask import Flask,render_template,request
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
   algor=request.values['predict_algor']
   algor_accuracy=algor+"_accuracy"
   pw= float(request.values['pw'])
   sw= float(request.values['sw'])
   pl= float(request.values['pl'])
   sl= float(request.values['sl'])
   test=[sl,sw,pl,pw]
   algorithm=model[str(algor)]
   output=algorithm.predict([test])[0]
   output=int(round(output,1))
   accuracy=model[str(algor_accuracy)]
   species={0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
   return render_template ('result.html',prediction_text="Entered details correspond to Flower Variety [-{}-]".format(species[output]), prediction_accuracy="Prediction Accuracy = {}".format(accuracy))

if __name__=='__main__':
    app.run(port=8000)