from flask import Flask


saga = Flask(__name__)


@saga.route('/')    ## this '/' basically means that it is an home page . and you can give the name that you need to route after the index (like /Home page)
def welcome():
    return "Welcome to the flask course ssssssss"

@saga.route("/Home")
def Home():
    return "Welcome to the home page"

if __name__ == "__main__":
    saga.run(debug=  True)  ## without this degub = True , if you mad any changes that will not reflect which you restart the system...