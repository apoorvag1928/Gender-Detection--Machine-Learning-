import numpy as np
#import PIL
#from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
#from sklearn.decomposition import PCA
#from sklearn.model_selection import train_test_split
import cv2
from sklearn.externals import joblib

def capimage(cap):

    if cap.isOpened():
        ret, frame=cap.read()
    #else:
    #    ret=False
        
        
    img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #plt.imshow(img)
    #plt.show()

    reduce_img= cv2.resize(img, (100, 100))

    plt.imshow(reduce_img)
    plt.show()
    
    return reduce_img


def main():
    """
    x=np.load("x.npy")
    y=np.load("y.npy")
    
    image=(x[3006].reshape(100,100))
    plt.imshow(image)

    x_test = x[25001:30000]
    y_test = y[25001:30000]
    
    x_test=(np.float32(x_test[:])/255.)
    """
 
    #x_test=pca.transform(x_test)

    classifier=joblib.load('trained_nn.pkl')
    pca=joblib.load('trained_nn_pca.pkl')
    #img= np.asarray(PIL.Image.open('text3.jpg'))
    #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img= cv2.resize(img, (100, 100))
    #print(img)
    
    cap=cv2.VideoCapture(0)
    
    while True:
        img=capimage(cap)
        #plt.imshow(img)
        #plt.show()
        img=img.reshape(1,-1)
        #print(img.shape)
        img=(np.float32(img)/255.)

        img=pca.transform(img)
        pred=classifier.predict(img)
        if pred==1:
            print("Male")
        else:
            print("Female")
            
        name=input("Press Y to take another image ")
        if name!='Y' and name!='y':
            break;
    cap.release()
            
            
    
if __name__ == "__main__":
    main()    

