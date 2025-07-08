from flask import Flask,render_template

app = Flask(__name__)

@app.route('/')
def go_to_index():
    return render_template('index.html')

@app.route('/gameball')  ## we can give this also /game
def go_to_ballgame():    ## we have to give this to html <div class="ballgame" onclick="window.location.href = '{{ url_for('go_to_ballgame') }}'">
    return render_template('game.html') 
@app.route('/rangame')  ## /randomgame   
##<div class="randomnumber" onclick="window.location.href = '{{ url_for('go_to_rannum') }}'">
def go_to_rannum():
    return render_template('randomgame.html')

if __name__ == "__main__":
    app.run(debug=True)