from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('cdv_model.pk1','rb'))
@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/result1',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        input_data = [int(request.form.get('age',False)),
                      int(request.form.get('gender',False)),
                      int(request.form.get('height',False)),
                      int(request.form.get('weight',False)),
                      int(request.form.get('BP',False)),
                      int(request.form.get('cholestrol',False)),
                      int(request.form.get('glucose',False)),
                      int(request.form.get('smoke',False)),
                      int(request.form.get('alcohol',False)),
                      int(request.form.get('work',False)),
                      int(request.form.get('family',False))]

        new_data = np.array(input_data).reshape(1, -1)

        # Make predictions using the ensemble classifier
        result_prediction = model.predict(new_data)
        if result_prediction==1:
            return render_template('result2.html')
        elif result_prediction==0:
            return render_template('result1.html')


if __name__ == '__main__':
    app.debug=True
    app.run()