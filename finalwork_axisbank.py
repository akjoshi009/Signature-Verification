from pylab import*
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure            
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import cv2
from pylab import rcParams
from scipy.misc import imread
import seaborn as sns; 
sns.set()



genuine_image_filenames = listdir("D:\\Data Science\\Challenges HackerEarth\\Axis_Bank\\Dataset\\dataset1\\real\\")
forged_image_filenames = listdir("D:\\Data Science\\Challenges HackerEarth\\Axis_Bank\\Dataset\\dataset1\\forge\\")
genuine_image_paths = "D:\\Data Science\\Challenges HackerEarth\\Axis_Bank\\Dataset\\dataset1\\real\\"
forged_image_paths = "D:\\Data Science\\Challenges HackerEarth\\Axis_Bank\\Dataset\\dataset1\\forge\\"
rcParams['figure.figsize'] = (8, 8)
count=0
col_names =  ['id','cy', 'cx','contours','contour','polar_contour','coords','coords_subpix','c_max_index','c_min_index','erosion','closing','opening','dilation','dist_2d']


#print(df)
def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return [rho, phi]
def cont(img):
        return max(measure.find_contours(img,.8), key=len)
i=0
def extract_features(image_path, vector_size=32):
    image = imread(image_path, mode="RGB")
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None

    return dsc

def Genratedata(genuine_image_filenames,filepath,typeX):
    df2  = pd.DataFrame(columns=range(2050))
    count=0
    for i in range(len(genuine_image_filenames)):
        id=genuine_image_filenames[count][5:-4]
        print(id)
    
    
    #img=preproc('D:\\Data Science\\Challenges//HackerEarth\\Axis_Bank\\sample_Signature\\genuine\\{0}'.format(genuine_image_filenames[count]))
    
    #plt.imshow(img,cmap='Set3')
    #plt.show()
        if(typeX==0):
            features=extract_features('D:\\Data Science\\Challenges HackerEarth\\Axis_Bank\\Dataset\\dataset1\\real\\{0}'.format(genuine_image_filenames[count]))
        else:
            features=extract_features('D:\\Data Science\\Challenges HackerEarth\\Axis_Bank\\Dataset\\dataset1\\forge\\{0}'.format(forged_image_filenames[count]))
            
        
        count=count+1;
    
        a=[None]*2050
        a[1]=id
        a[0]=typeX
    
        for j in range(0,2048):
         #print(features[j])  
             a[j+2]=features[j]
         
        
        
    
        df2.loc[i]=a
        
    return df2    
        #df.to_csv("D:\\Data Science\\Challenges HackerEarth\\Axis_Bank\\sample_Signature\\traincs.csv", encoding='utf-8', index=False)



gendata=Genratedata(genuine_image_filenames,genuine_image_paths,0)
forgdata=Genratedata(forged_image_filenames,forged_image_paths,1)
df = pd.concat([gendata, forgdata], ignore_index=True)
 ## key→old name, value→new name
df.columns = ['Userid' if x==1 else 'Label' if x==0 else x for x in df.columns]
print(df.columns)
df.to_csv("D:\\Data Science\\Challenges HackerEarth\\Axis_Bank\\sample_Signature\\traincs.csv", encoding='utf-8', index=False)
X_train, X_test, Y_train, Y_test = train_test_split(df[df.columns[2:]],df["Label"], test_size=0.2, random_state=0)
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=100,
                                           multi_class='multinomial')
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('logistic', logistic)])

# #############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.

rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000
ax = sns.heatmap(rbm_features_classifier)
# Training RBM-Logistic Pipeline
rbm_features_classifier.fit(X_train,Y_train)

# Training the Logistic regression classifier directly on the pixel
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.
raw_pixel_classifier.fit(X_train, Y_train)

# #############################################################################
# Evaluation

Y_pred = rbm_features_classifier.predict(X_test)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(Y_test, Y_pred)))

Y_pred = raw_pixel_classifier.predict(X_test)
print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(Y_test, Y_pred)))

# #############################################################################
# Plotting

plt.figure(figsize=(4, 4.3))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((1, 2048)), cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()

