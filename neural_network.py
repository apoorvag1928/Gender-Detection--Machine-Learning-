import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
#from sklearn.model_selection import train_test_split
#import cv2
from sklearn.externals import joblib


def main():
    x=np.load("x.npy")
    y=np.load("y.npy")

    image=(x[1000].reshape(100,100))
    plt.imshow(image)

    #x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25)
    x_train = x[0:25000]
    x_test = x[30001:]
    y_train = y[0:25000]
    y_test = y[30001:]
    
    x_train=(np.float32(x_train[:])/255.)
    x_test=(np.float32(x_test[:])/255.)

    pca=PCA(n_components=900)
    pca.fit(x_train)
    joblib.dump(pca, 'trained_nn_pca.pkl')

    x_train=pca.transform(x_train)
    x_test=pca.transform(x_test)

    classifier=MLPClassifier(hidden_layer_sizes=(104,))

    classifier.fit(x_train, y_train)
    joblib.dump(classifier, 'trained_nn.pkl')

    pred=classifier.predict(x_test)

    ac=accuracy_score(y_test, pred)

    print("SVM accuracy: ", ac*100)
    
if __name__ == "__main__":
    main()    