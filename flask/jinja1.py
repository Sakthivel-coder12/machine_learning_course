## going to get the input from the user and dispaly it the html file itself 

from flask import Flask,render_template,request,redirect,url_for

app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('index.html')
@app.route('/login')
def login():
    return render_template('login5.html')
@app.route('/submit',methods = ["GET","POST"])
def submit():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
    return render_template('index.html',results = (username,password))
if __name__ == "__main__":
    app.run(debug=True)