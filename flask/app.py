from flask import Flask
## This is create the instance of the class , whcih will be our wsgt (web server gateway interface)
app = Flask(__name__)
@app.route('/')
def welcome():
    return "Welcome to the flask...!"
if __name__ == "__main__":
    app.run() # This is the command that used to run the flask program 
## This is the basic skleton that we need to create for every flask program 