from flask import Flask, request, jsonify,render_template
import os
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from VideoSync import STARTVideo
from wsgiref import simple_server

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    f = request.files['file']
    filename = secure_filename(f.filename)
    filepath = os.path.join("static/uploads", filename)
    f.save(filepath)
    result = STARTVideo(filepath)
    return render_template("uploaded.html", result=result, filepath=filepath)


#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    port = int(os.getenv("PORT"))
    host = '0.0.0.0'
    httpd = simple_server.make_server(host=host,port=port, app=app)
    httpd.serve_forever()