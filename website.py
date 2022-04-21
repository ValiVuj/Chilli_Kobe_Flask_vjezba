X= "Chilli"
Y="Kobe"

sampleX="static/Chilli.jpg"
sampleY="static/Kobe.jpg"

#category where I will keep users uploads
UPLOAD_FOLDER = "static/uploads"
#allowed files
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

import os
from turtle import width
import numpy as np

from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from keras.preprocessing import image
from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf

#create the website object
app = Flask(__name__)

def load_model_from_file():
    #setting up the machine learning session
    mySession = tf.compat.v1.Session()
    set_session(mySession)
    myModel = load_model("best_model.pb")
    myGraph=tf.compat.v1.get_default_graph()
    return(mySession,myModel,myGraph)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#define the view for the top level page
@app.route("/", methods=["GET", "POST"])
def upload_file():
    #initial webpage load
    if request.method =="GET":
        return render_template("index.html", myX=X, myY=Y, mySampleX=sampleX, mySampleY=sampleY)
    else: #if request.method =="post"
        #check if the post request has the file
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        #if user does not select file. browser may also
        #submit an empty part without filename
        if file.filename=="":
            flash("No selected file")
            return redirect(request.url)
        #if it doesnt look lik an image file
        if not allowed_file(file.filename):
            flash("I only accept files of type" +str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        #when user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("uploaded_file", filename=filename))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    test_image = image.load_img(UPLOAD_FOLDER+"/"+filename, target_size=(150,150))
    test_image= image.img_to_array(test_image)
    test_image=np.expand_dims(test_image, axis=0)

    mySession = app.config["SESSION"]
    myModel = app.config["MODEL"]
    myGraph = app.config["GRAPH"]
    print("loaded")

    with myGraph.as_default():
        set_session(mySession)
        result=myModel.predict(test_image) #ovdje radimo predikciju, rezultat 0 ili 1
        image_src="/"+UPLOAD_FOLDER +"/"+filename
        if result(0) < 0.5:
            answer = "<div class='col text-center'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+X+" "+str(result[0])+"</h4></div><div class='col'></div><div class='w-100'></div>"     
        else:
            answer = "<div class='col'></div><div class='col text-center'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+Y+" "+str(result[0])+"</h4></div><div class='w-100'></div>"     
        results.append(answer)
        return render_template('index.html',myX=X,myY=Y,mySampleX=sampleX,mySampleY=sampleY,len=len(results),results=results)





def main():
    (mySession, myModel, myGraph) = load_model_from_file()

    app.config["SECRET_KEY"] = "tajni kljuc"

    app.config["SESSION"] = mySession
    app.config["MODEL"] = myModel
    app.config["GRAPH"] = myGraph

    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = 16*1024*1024
    app.run()

results=[]

main()







