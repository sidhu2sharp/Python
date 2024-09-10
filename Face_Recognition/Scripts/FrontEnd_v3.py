import numpy as np
import cv2
import random
from io import StringIO
# from os.path import isfile

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from keras import applications

# from keras.models import Model
from keras.models import load_model

import streamlit as st


def loadData():
    # if not isfile("C:/Programming/Python/Face_Recognition/TrainData/trainImages.npy") & isfile(
    #     "C:/Programming/Python/Face_Recognition/TrainData/trainLabels.npy"
    # ):
    #     getCleanedData()
    X = np.load(
        "C:/Programming/Python/Face_Recognition/TrainData/trainImages.npy", allow_pickle=True
    )
    y = np.load(
        "C:/Programming/Python/Face_Recognition/TrainData/trainLabels.npy", allow_pickle=True
    )
    X, y = shuffle(X, y, random_state=0)
    return X, y


def loadModel(myModel="Xception"):
    if myModel == "Xception":
        model = load_model(
            "C:/Programming/Python/Face_Recognition/Models/xception_transfer_learning_model.keras"
        )
        return model
    elif myModel == "InceptionV3":
        model = load_model(
            "C:/Programming/Python/Face_Recognition/Models/inceptionv3_transfer_learning_model.keras"
        )
        return model
    else:
        return None


def preProcessTestImage(imgFile, col):
    imgFile = "C:/Programming/Python/Face_Recognition/TestData/" + imgFile
    img = cv2.imread(imgFile, 1)
    img = cv2.resize(img, (256, 256))
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    col.image(img1)
    img = applications.xception.preprocess_input(img.astype("float64"))
    # col.info(model.predict(np.expand_dims(img, axis=0)).max())
    confList = model.predict(np.expand_dims(img, axis=0))
    conf = confList.max()
    confIndex = np.argmax(confList)
    return conf, confIndex


def displayInputImages(X, y, column2):
    with column2:
        disImages = st.container(border=True)
        with disImages.caption(":blue[Input Images]"):
            col1, col2, col3 = disImages.columns(3)
            X, y = shuffle(X, y, random_state=0)
            for i in range(5):
                for col, i in zip([col1, col2, col3], range(3)):
                    val = random.randint(0, y.shape[0])
                    img = cv2.cvtColor(X[val], cv2.COLOR_BGR2RGB)
                    col.image(img, caption=y[val] + " " + str(img.shape))


def displaySourceCode(column1):
    with column1:
        disFiles = st.expander(":blue[Source Code]", expanded=False)
        with disFiles.caption("Code"):
            uploaded_file = st.file_uploader(":blue[Choose a file]")
            if uploaded_file is not None:
                # To read file as string:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                string_data = stringio.read()
                #     disFiles.text(string_data)
                disFiles.code(string_data, language="python")
    return


X, y = loadData()

le = LabelEncoder()
names = le.fit_transform(y)

st.set_page_config(layout="wide")

titleContainer = st.container(border=True)
titleContainer.title("Face Recognition")

column1, column2 = st.columns([4, 3])

with st.sidebar.header("Input Panel", divider="gray"):
    st.sidebar.button("Input Images", on_click=displayInputImages(X, y, column2))

with st.sidebar.header("Selection Panel", divider="gray"):
    displaySourceCode(column1)

with st.sidebar:
    option = st.sidebar.selectbox(
        "Model Selection", ("Choose Model", "Xception", "InceptionV3")
    )
    modelContainer = column1.container(border=True)
    with modelContainer.caption(":blue[Model Details]"):
        with st.status("Loading model ..."):
            expander = modelContainer.expander("See Architecture")
            if option == "Xception":
                model = loadModel(option)
                layers = len(model.layers)
                modelContainer.info(
                    "Loaded: " + option + " model with " + str(layers) + " layers"
                )
                expander.image(
                    "C:/Programming/Python/Face_Recognition/Architecture/The-Xception-algorithm-diagram.png"
                )
            elif option == "InceptionV3":
                model = loadModel(option)
                layers = len(model.layers)
                modelContainer.info(
                    "Loaded: " + option + " model with " + str(layers) + " layers"
                )
                expander.image(
                    "C:/Programming/Python/Face_Recognition/Architecture/InceptionV3-Architecture.png"
                )
            else:
                modelContainer.info("No model selected!")

col1, col2 = column1.columns(2)

org = col1.container(border=True)
pred = col2.container(border=True)

with org:
    uploadedTestImage = st.file_uploader(":blue[Choose test image]")
    if uploadedTestImage is not None:
        # with st.spinner("Processing the image and making prediction . . ."):
        with st.status("Processing the image and making prediction ..."):
            pred.caption(":blue[Model Prediction]")
            # pred.divider()
            lbl = uploadedTestImage.name.split(".")[1]
            lbl = lbl.split("_")
            lbl = " ".join(lbl).title()
            conf, confIndex = preProcessTestImage(uploadedTestImage.name, org)
            myname = le.inverse_transform([confIndex])
            myname = myname[0].split("_")
            myname = " ".join(myname).title()
            pred.metric(":red[Original: ]", value=lbl)
            pred.metric(":red[Predicted: ]", value=myname)
            pred.metric(
                ":red[Confidence Level: ]", value=str(round(conf * 100, 2)) + "%"
            )
        if lbl == myname:
            pred.success(" Successful prediction!", icon="✅")
        else:
            pred.error(" Failed prediction!", icon="❌")
