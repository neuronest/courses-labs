import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def show(image):
    plt.imshow(np.squeeze(image.astype("uint8")), cmap="gray")
    plt.title("image shape: "+ str(image.shape), fontsize=14)
    plt.axis('off');
    
def show_multiple(images, figsize):
    fig, ax = plt.subplots(ncols=len(images), figsize=figsize)
    for col, image in zip(ax, images):
        col.imshow(np.squeeze(image.astype("uint8")), cmap="gray")
        col.set_title("image shape: "+ str(image.shape), fontsize=14)
    plt.tight_layout()
    plt.axis('off');


def get_splitted_data_with_size(image_size, sample_size, test_ratio, classes, seed):
    X, Y = [], []
    for label, animal in enumerate(classes):
        files = os.listdir(os.path.join('data', animal))
        random.shuffle(files)
        files = files[:(sample_size // len(classes))]
        for file in files:
            img = load_img(os.path.join('data', animal, file), 
                           target_size=image_size)
            X.append(img_to_array(img))
            Y.append(label)
    return train_test_split(np.asarray(X), np.asarray(Y), test_size=test_ratio, random_state=seed)

def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric, name in zip(axs, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(
            range(1, len(model_history.history[metric]) + 1), 
            model_history.history[metric]
        )
        ax.plot(
            range(1, len(model_history.history['val_' + metric]) + 1), 
            model_history.history['val_' + metric]
        )
        ax.set_title('Model ' + name)
        ax.set_ylabel(name)
        ax.set_xlabel('Epoch')
        ax.legend(['train', 'val'], loc='best')
    plt.show()
    
def scale_data(X_tr, X_val, return_scaler=False):
    shape_tr, shape_val = X_tr.shape, X_val.shape
    X_tr_flat = np.ravel(X_tr).reshape(-1, 1)
    X_val_flat = np.ravel(X_val).reshape(-1, 1)
    min_max_scaler = MinMaxScaler()
    X_tr_scaled = min_max_scaler.fit_transform(X_tr_flat).reshape(shape_tr)
    X_val_scaled = min_max_scaler.transform(X_val_flat).reshape(shape_val)
    if not return_scaler:
        return X_tr_scaled, X_val_scaled
    else:
        return X_tr_scaled, X_val_scaled, min_max_scaler
    
def apply_scaling(X, scaler):
    shape_X = X.shape
    X_flat = np.ravel(X).reshape(-1, 1)
    X_scaled = scaler.transform(X_flat).reshape(shape_X)
    return X_scaled
