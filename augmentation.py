import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import  Sequential, Model
import matplotlib.pyplot as plt
import os
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
#from sklearn import metrics
from sklearn.metrics import roc_curve, auc 



directory_train ="divided_data/train"
directory_val ="divided_data/val"
directory_test ="divided_data/test"

batch_size = 16

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory_train,
    target_size=(150, 150),  
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42)


validation_generator = test_datagen.flow_from_directory(
    directory_val,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = False)

test_generator = test_datagen.flow_from_directory(
    directory_test,
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)

target_labels = next(os.walk(directory_train))[1]
target_labels.sort()
#print(target_labels)

num_classes=len(target_labels)
#print(num_classes)

batch= next(train_generator)
batch_images = np.array(batch[0])
#print(batch_images)
batch_labels = np.array(batch[1])
#print(batch_labels)



target_labels=np.asarray(target_labels)

plt.subplots(figsize=(15, 10))
for n, i in enumerate(range(6)):
    ax=plt.subplot(3,3,n+1)
    plt.imshow(batch_images[i])
    plt.title(target_labels[np.argmax(batch_labels[i])])
    plt.axis("off")
#plt.show()

taken_model = tf.keras.applications.ResNet50V2(weights="imagenet")
#print(taken_model)

taken_model_input = taken_model.input
#print(taken_model_input)

taken_model_output = taken_model.layers[-2].output
#print(taken_model_output)


model = tf.keras.Model(inputs=taken_model_input, outputs=taken_model_output)
#print(model.summary())

def add_your_own_head(your_own_model):
    model= Sequential([
        your_own_model,
        Dense(32,activation="relu"),
        Dropout(0.5),
        Dense(2, activation="sigmoid")
    ])

    return model

imported_model = add_your_own_head(model)
#imported_model.summary()


model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(2, activation='sigmoid')])

imported_model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

filepath = "best_model.pb"

my_callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
    keras.callbacks.ModelCheckpoint(filepath,
                                    save_best_only=True,
                                    monitor="val_accuracy",
                                    mode="max"),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                      factor=0.1,
                                      patience=3,
                                      cooldown=1)
]

# history = imported_model.fit(
#     train_generator,
#     steps_per_epoch = train_generator.samples // train_generator.batch_size,
#     validation_data = validation_generator,
#     #validation_steps = validation_generator.samples // batch_size,
#     callbacks = my_callbacks,
#     epochs=100)

# score = imported_model.evaluate(test_generator, batch_size=1, steps=test_generator.samples)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])


model = keras.models.load_model("best_model.pb")  #ovo je za loadanje vec pretreniranog modela, 
print(model)

# plt.figure(figsize=(15,5))
# plt.subplot(121)
# plt.plot(imported_model.history['accuracy'])
# plt.plot(imported_model.history['val_accuracy'])
# plt.title('Accuracy vs. epochs')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training', 'Validation'], loc='lower right')

# plt.subplot(122)
# plt.plot(imported_model.history['loss'])
# plt.plot(imported_model.history['val_loss'])
# plt.title('Loss vs. epochs')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training', 'Validation'], loc='upper right')
# plt.show()

# if os.path.isfile("C:/Users/Valentina/Desktop/vjezba_za_projekt/model/Kobe_vs_Chilli.pb") is False:
#     model.save("C:/Users/Valentina/Desktop/vjezba_za_projekt/model/Kobe_vs_Chilli.pb") ovo je za sejvanje modela, ili mozemo staviti u keras.callbacks.ModelCheckpoint, sa ovim .save sejva tezine, arhitekuru modela, train konfiguraciju i stanje optimajzera



# preds = model.predict(validation_generator, verbose=1)

# fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color="darkorange",
# lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)

# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver operating characteristic example")
# plt.legend(loc="lower right")
# plt.show()
