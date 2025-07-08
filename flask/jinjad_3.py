from flask import Flask, request, url_for, redirect, render_template

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('form.html')

@app.route("/success/<int:score>")
def success(score):
    res = ""
    if score <= 50:
        res = "FAIL"
    else:
        res = "PASS"
    return render_template('form.html', results=res)

@app.route("/fail/<int:score>")
def fail(score):
    return render_template('form.html', results=(score, "fail"))

@app.route('/submit', methods=['POST'])
def submit():
    total = 0
    if request.method == "POST":
        cs = float(request.form['com'])
        c = float(request.form['c'])
        ml = float(request.form['ml'])
        dl = float(request.form['dl'])
        total = (cs + c + ml + dl) / 4
    return redirect(url_for('success', score=int(total)))

if __name__ == "__main__":
    app.run(debug=True)
