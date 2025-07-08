## Building Url dynamically
## variable Rule
## jinja 2 templete

from flask import Flask,render_template,redirect,url_for,request

app = Flask(__name__)

@app.route('/')
def welcome():
    return "<h1>Welcome to the sakthivel web-site ❤️❤️</h1>"

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/login',methods = ['GET','POST'])
def login():
    return render_template('login5.html')
@app.route('/submit/<score>',methods = ['GET','POST'])   ## when you click on the submit this route will get executed 
def submit(score):
    if request.method == 'POST':
        uname = request.form['username']
        password = request.form['password']
    return "The result is " + score
@app.route('/submit1/<int:score>')
def success(score):
    res = ""
    if score >= 50:
        res = "pass"
    else:
        res = "fail"
    return "The result is " + res
## This si going to dispaly the result int the html file itself
@app.route('/success/<int:score>')
def success1(score):
    res = ""
    if score >= 50:
        res = "pass"
    else:
        res = "fail"
    return render_template('result.html',results = res)

@app.route('/success1/<int:score>')
def success2(score):
    res = ""
    if score >= 50:
        res = "pass"
    else:
        res = "fail"
    exp = {"score":score,"res":res}
    return render_template('result1.html',results = exp)
@app.route('/success2/<int:score>')
def success3(score):
    return render_template('result2.html',results = score)
if __name__ == "__main__":
    app.run(debug=True)