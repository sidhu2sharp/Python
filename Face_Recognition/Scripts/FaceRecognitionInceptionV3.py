import matplotlib.pyplot as plt
import numpy as np
import cv2, random
# import pywt, pywt.data
from os import listdir
from os.path import isfile, isdir, join, dirname
from tqdm import tqdm
from yuface import detect

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

# from sklearn.utils import shuffle
from keras.layers import Dense, GlobalAveragePooling2D
from keras import applications
from keras.models import Model
from keras.models import load_model
# from keras.optimizers import Adam


class FR:
    def __init__(self):
        self.DIR = "C:/Programming/Python/Face_Recognition"
        self.RAWIMAGEDIR = "C:/Programming/Python/Face_Recognition/images"
        self.imgFileList = []  # Full path of the image filenames
        self.croppedfaceImgList = []  # Cropped face images
        self.labelList = []  # Image labels
        self.trainImages = []
        return

    def getDirList(self) -> None:
        datafile = [
            f for f in listdir(self.RAWIMAGEDIR) if isdir(join(self.RAWIMAGEDIR, f))
        ]
        return datafile

    def getImageFiles(self) -> None:
        myDirs = self.getDirList()
        for dir in tqdm(myDirs):
            # datafile1 = [self.RAWIMAGEDIR + '/' + dir + '/' + f for f in listdir(self.RAWIMAGEDIR + '/' + dir) if isfile(join(self.RAWIMAGEDIR + '/' + dir, f))]
            for f in listdir(self.RAWIMAGEDIR + "/" + dir):
                if isfile(join(self.RAWIMAGEDIR + "/" + dir, f)):
                    self.imgFileList.append(self.RAWIMAGEDIR + "/" + dir + "/" + f)
                    image = cv2.imread(self.imgFileList[-1])
                    image = cv2.resize(image, (256, 256))
                    self.trainImages.append(image)
                    self.labelList.append(dir)
        return

    def getCroppedface(self, img: np.array) -> np.array:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bboxes, _ = detect(img, conf=0.5)
        if bboxes.size != 0:
            bbox = bboxes[0]
            img1 = img[bbox[1] : (bbox[1] + bbox[3]), bbox[0] : (bbox[0] + bbox[2])]
            return img1
        else:
            return np.array([])

    # def waveletTransformation(self, img: np.array) -> np.array:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     coeffs2 = pywt.dwt2(img, "bior1.3")
    #     return coeffs2

    def getCroppedfaceImageList(self) -> None:
        self.getImageFiles()
        for imgfile in tqdm(self.imgFileList):
            img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (256, 256))
            img = self.getCroppedface(img)
            # img = cv2.resize(img, (256, 256))
            if img is not None:
                if img.size > 5000:
                    img = cv2.resize(img, (256, 256))
                    self.croppedfaceImgList.append(img)
                    self.trainImages.append(img)
                    self.labelList.append(dirname(imgfile).split("/")[-1])
                    # Wavelet Transformation
                    # try:
                    #     LL, (LH, HL, HH) = self.waveletTransformation(img)
                    #     for pwImg in [LL, LH, HL, HH]:
                    #         pwImg = cv2.resize(pwImg, (256, 256))
                    #         pwImg = cv2.merge((pwImg, pwImg, pwImg)).astype(np.uint)
                    #         self.trainImages.append(pwImg)
                    #         self.labelList.append(dirname(imgfile).split('/')[-1])
                    # except:
                    #     continue
        return

    def getCleanedData(self) -> None:
        self.getCroppedfaceImageList()
        trainImages = "C:/Programming/Python/Face_Recognition/TrainData/trainImages.npy"
        trainLabels = "C:/Programming/Python/Face_Recognition/TrainData/trainLabels.npy"
        np.save(trainImages, np.array(self.trainImages))
        np.save(trainLabels, np.array(self.labelList))
        return


if __name__ == "__main__":
    fr = FR()
    if not isfile("C:/Programming/Python/Face_Recognition/TrainData/trainImages.npy") & isfile(
        "C:/Programming/Python/Face_Recognition/TrainData/trainLabels.npy"
    ):
        fr.getCleanedData()
    X = np.load(
        "C:/Programming/Python/Face_Recognition/TrainData/trainImages.npy", allow_pickle=True
    )
    y = np.load(
        "C:/Programming/Python/Face_Recognition/TrainData/trainLabels.npy", allow_pickle=True
    )
    X, y = shuffle(X, y, random_state=0)
    print(X.shape, y.shape)

    # Random check of images
    for i in range(2):
        val = random.randint(0, y.shape[0])
        img = cv2.cvtColor(X[val], cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(y[val])
        plt.show()

    print("Pre-processing data . . .")
    X = applications.inception_v3.preprocess_input(X.astype("float64"))
    #  Label encode the categorical output
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Checking if data is normalized to between -1 to +1
    X.min(), X.max()

    # Random image check
    num = random.randint(0, y.shape[0])
    img = (X[num] + 1) / 2
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.title(le.inverse_transform([y[num]])[0])
    plt.show()

    if not isfile(
        "C:/Programming/Python/Face_Recognition/Models/inceptionv3_transfer_learning_model.keras"
    ):
        # First Train the top layers (which we have added)
        num_classes = len(list(set(y)))
        pre_trained_layer = applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=(X.shape[1], X.shape[2], X.shape[3]),
        )
        x = pre_trained_layer.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=pre_trained_layer.input, outputs=predictions)
        for layer in pre_trained_layer.layers:
            layer.trainable = False
        # 'categorical_crossentropy' works on one-hot encoded target, while 'sparse_categorical_crossentropy' works on integer target. Here we are using Label Encoder,
        #  hence the 'sparse_categorical_crossentropy'
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        print(model.summary(show_trainable=True))
        model.fit(
            X,
            y,
            validation_split=0.25,
            shuffle=True,
            epochs=25,
            batch_size=32,
            verbose=1,
        )

        # Now choose the layer from the pre-trained network which we want to un-freeze in addition to the top layers and re-train.
        for layer in model.layers[:100]:
            layer.trainable = False
        for layer in model.layers[100:]:
            layer.trainable = True

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        print(model.summary(show_trainable=True))
        model.fit(
            X,
            y,
            validation_split=0.15,
            shuffle=True,
            epochs=25,
            batch_size=32,
            verbose=1,
        )

        # Instead of unfreezing the entire model, we unfreezed the top 100 layers in the above code.
        # # Unfreeze the base_model. Note that it keeps running in inference mode
        # # since we passed `training=False` when calling it. This means that
        # # the batchnorm layers will not update their batch statistics.
        # # This prevents the batchnorm layers from undoing all the training
        # # we've done so far. Ref: https://keras.io/guides/transfer_learning/
        # for layer in pre_trained_layer.layers:
        #         layer.trainable = True
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])
        # print (model.summary(show_trainable='True'))
        # model.fit(X, y, validation_split=0.15, shuffle=True, epochs=10, batch_size=32, verbose=1)

        model.save(
            "C:/Programming/Python/Face_Recognition/Models/inceptionv3_transfer_learning_model.keras"
        )
    else:
        model = load_model(
            "C:/Programming/Python/Face_Recognition/Models/inceptionv3_transfer_learning_model.keras"
        )
        print(model.summary(show_trainable=True))

    # Evaluate the model
    score = model.evaluate(X, y, verbose=1)
    print(
        "\n%s: %.2f%%" % ("Model " + str.title(model.metrics_names[1]), score[1] * 100)
    )

    # Testing

    TEST_DIR = "C:/Programming/Python/Face_Recognition/TestData/"
    Total_Count = 0
    Pass_Count = 0
    for f in listdir(TEST_DIR):
        if isfile(join(TEST_DIR, f)):
            label = f.split(".")[1]
            img = cv2.imread(join(TEST_DIR, f), 1)
            img = cv2.resize(img, (256, 256))
            cropped = fr.getCroppedface(img)
            cropped = cv2.resize(cropped, (256, 256))
            # plt.imshow(cropped)
            # plt.title(f'{label}')
            # plt.show()
            # Check if decision can be made with cropped image for better accuracy
            X = applications.xception.preprocess_input(cropped.astype("float64"))
            pred = model.predict(np.expand_dims(X, axis=0), verbose=0).max()
            if pred > 0.9:  # Image identified
                pred = le.inverse_transform(
                    [np.argmax(model.predict(np.expand_dims(X, axis=0), verbose=0))]
                )[0]
                b, g, r = cv2.split(img)
                img = cv2.merge([r, g, b])
                plt.imshow(img)
                plt.title(f"{label}")
                plt.axis("off")
                plt.show()
                confidence = round(
                    model.predict(np.expand_dims(X, axis=0), verbose=0).max() * 100, 2
                )
                print(f"Original: {label} - Predicted: {pred} ({confidence}%)")
                Total_Count += 1
                if label == pred:
                    Pass_Count += 1
            else:
                # Check if decision can be made with original image
                X = applications.xception.preprocess_input(img.astype("float64"))
                pred = model.predict(np.expand_dims(X, axis=0), verbose=0).max()
                if pred > 0.7:  # Image identified
                    pred = le.inverse_transform(
                        [np.argmax(model.predict(np.expand_dims(X, axis=0), verbose=0))]
                    )[0]
                    b, g, r = cv2.split(img)
                    img = cv2.merge([r, g, b])
                    plt.imshow(img)
                    plt.title(f"{label}")
                    plt.axis("off")
                    plt.show()
                    confidence = round(
                        model.predict(np.expand_dims(X, axis=0), verbose=0).max() * 100,
                        2,
                    )
                    print(f"Original: {label} - Predicted: {pred} ({confidence}%)")
                    Total_Count += 1
                    if label == pred:
                        Pass_Count += 1
                else:
                    b, g, r = cv2.split(img)
                    img = cv2.merge([r, g, b])
                    plt.imshow(img)
                    plt.title(f"{label}")
                    plt.axis("off")
                    plt.show()
                    pred1 = model.predict(np.expand_dims(X, axis=0), verbose=0)[0]
                    # Incase of indecision, get the best two predictions
                    pred = np.argpartition(pred1, -2)[-2:]
                    print(
                        f"Original: {label} - Predicted: {le.inverse_transform([pred[1]])[0]} ({round(pred1[pred[1]]*100, 2)}%) or {le.inverse_transform([pred[0]])[0]} ({round(pred1[pred[0]]*100, 2)}%)"
                    )
                    Total_Count += 1
    print(
        f"Accuracy: Out of a total of {Total_Count} images, {Total_Count - Pass_Count} were misclassified.\n%Pass: {round((Pass_Count / Total_Count) * 100, 2)}%"
    )
