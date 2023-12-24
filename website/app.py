from flask import Flask, render_template, request, send_from_directory
import os

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))

app.config["upload_folder"] = os.path.join(script_dir, "uploads")

''' 
------------SETUP------------ 
export FLASK_APP=app
export FLASK_ENV=development
'''
'''
def predict(model, img):
    model.forward(img) etc. etc.
    pred = argmax
    if pred[0] == "No Finding":
        return "The model is {.2f}% sure that this image isn't cancerous".format(pred[0])
    else:
        return "The model is {.2f}% sure that the image is cancerous, with a {.2f}% surety of {}".format(1-NoFinding, pred[0], class_names[class_index])
'''
@app.route('/')
def index():
    #unpickle model
    #pred = predict(model, img)
    pred = "Please insert an image"
    return render_template('index.html', content=pred)

@app.route('/about')
def about():
    return render_template('/about.html')

@app.route('/news')
def news():
    return render_template('/test.html')

@app.route('/', methods=['POST'])
def upload_image():
    if "file" not in request.files:
        return render_template("index.html", content="No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", content="No file uploaded")

    filename = os.path.join(app.config["upload_folder"], file.filename)
    file.save(filename)
    return render_template("index.html", content="File successfully uploaded", filename=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["upload_folder"], filename)

if __name__ == "__main__":
    app.run(debug=True)
