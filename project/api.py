from flask import Flask,render_template,request,url_for
from model.en_tech import predict_diabetes

app = Flask(__name__)

@app.route("/",methods = ['GET','POST'])
def welcome():
    hard_result,soft_result = 0,0
    if request.method == 'POST': 
        try:
            pregnancies = request.form['pregnancies']
            glucose = request.form['glucose']
            bloodPressure = request.form['bloodPressure']
            skinThickness = request.form['skinThickness']
            insulin = request.form['insulin']
            bmi = request.form['bmi']
            dpf = request.form['dpf']
            age = request.form['age']
            res = [pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age]
            hard_result, soft_result = predict_diabetes(res)
        except Exception as e:
            print(f"Error: {e}")
    return render_template("form.html",hard = hard_result,soft = soft_result)

if __name__ == "__main__":
    app.run(debug=True)