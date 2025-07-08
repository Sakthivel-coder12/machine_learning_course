from flask import Flask,render_template,request

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/login' ,methods = ['GET','POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        password = request.form['password']
        return f"The username  : {uname} \n The password : {password}"
    return render_template('login5.html')

@app.route('/submit',methods = ['GET','POST'])   ## when you click on the submit this route will get executed 
def submit():
    if request.method == 'POST':
        uname = request.form['username']
        password = request.form['password']
        return f"The username  : {uname} \n The password : {password}"
    return render_template('login5.html')
if __name__ == "__main__":
    app.run(debug=True)




# int the above cause the both login route and the submit route have the access to read and dispaly the values 
# in the form , because both have the post method to access , if we don't what that kind of thing then we need to 
# create a login route with render_template('/login5.html') only

'''
@app.route('/login')
def login():
    return render_template('login5.html')

@app.route('/submit', methods=['POST'])
def submit():
    uname = request.form['username']
    password = request.form['password']
    return f"The username: {uname} <br> The password: {password}"

'''