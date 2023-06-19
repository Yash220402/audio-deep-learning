import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = ""

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    
    return X, y
    

def plot_history(history):
    
    fig, axs = plt.subplots(2)
    
    # accuracy plot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Eval")
    
    # error plot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Eval")
    
    plt.show()
    
def prepare_datasets(test_size, validation_size):
    X, y = load_data(DATA_PATH)
    pass 
    
    
def build_model(input_shape):
    model = keras.sequential
    
    # 1st conv layer 
    model.add(keras.layers.Conv2d(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPooling2d((3,3), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    # 2st conv layer 
    model.add(keras.layers.Conv2d(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPooling2d((3,3), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    # 3st conv layer 
    model.add(keras.layers.Conv2d(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPooling2d((3,3), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    # Flatten out and feed it into the dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    
    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))
    
    return model
    

def predict(model, X, y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    print("Target: {}, Predicted label: {}".format(y, predicted_index))
    

if __name__ == "__main__":
    
    