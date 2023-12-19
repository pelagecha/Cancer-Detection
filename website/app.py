from flask import Flask, render_template

app = Flask(__name__)

''' 
------------SETUP------------ 
export FLASK_APP=app
export FLASK_ENV=development
-------------RUN-------------
flask run
'''

@app.route('/')
def index():
    return render_template('index.html')
