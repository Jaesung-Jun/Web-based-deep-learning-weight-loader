import numpy as np
import PIL as Image
from keras.utils import np_utils
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image

def pre_processing(dir, label):
    
    X_output = []
    Y_output = []

    file_dir = [f for f in listdir(dir) if isfile(join(dir, f))]
    file_dir.sort()
    prefix_text = "Data Preprocessing from " + dir + " : "
    __printProgressBar(0, len(file_dir), prefix=prefix_text, suffix = "Complete", length = 50)

    for i in range(0, len(file_dir)):

        inputImage = dir + "/" + file_dir[i]

        im = Image.open(inputImage)
        X_output.append(np.array(im).tolist())

        for j in range(0, len(label)):
            if label[j] in inputImage:
                Y_output.append(j)

        __printProgressBar(i+1, len(file_dir), prefix=prefix_text, suffix = "Complete", length = 50)

    return X_output, Y_output

def Data_All_Preprocessing(dir, label, X_size=224, Y_size=224):

    """

    Arguments

    ● label = tuple of labels
    ● dir = target directory
    ● X_size = image X(width) size
    ● Y_size = image Y(height) size

    """
    X_temp = []
    Y_temp = []

    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    X_val = []
    Y_val = []

    ########Train Data########
    for i in range(0, len(listdir(dir))):

        X_temp, Y_temp = pre_processing(dir=(dir + "/" + listdir(dir)[i]) + "/train data", label=label)

        for i in range(0, len(X_temp)):
            X_train.append(X_temp[i])

        for i in range(0, len(Y_temp)):
            Y_train.append(Y_temp[i])

    ########Test Data########
    for i in range(0, len(listdir(dir))):

        X_temp, Y_temp = pre_processing(dir=(dir + "/" + listdir(dir)[i]) + "/test data", label=label)

        for i in range(0, len(X_temp)):
            X_test.append(X_temp[i])

        for i in range(0, len(Y_temp)):
            Y_test.append(Y_temp[i])

    ########Validation Data########  
    for i in range(0, len(listdir(dir))):        

        X_temp, Y_temp = pre_processing(dir=(dir + "/" + listdir(dir)[i]) + "/val data", label=label)

        for i in range(0, len(X_temp)):
            X_val.append(X_temp[i])

        for i in range(0, len(Y_temp)):
            Y_val.append(Y_temp[i])

    print("Data Regularizing........", end='\r')
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    Y_val = np.array(Y_val)

    X_train = X_train.reshape(X_train.shape[0], X_size, Y_size, 3).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], X_size, Y_size, 3).astype('float32') / 255
    X_val = X_val.reshape(X_val.shape[0], X_size, Y_size, 3).astype('float32') / 255

    Y_train = np_utils.to_categorical(Y_train, len(label))
    Y_test = np_utils.to_categorical(Y_test, len(label))
    Y_val = np_utils.to_categorical(Y_val, len(label))

    print("Preprocessing All Complete!")

    return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)

def __printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

"""
x_train = []
x_test = []
x_val = []

y_train = []
y_test = []
y_val = []

np.array(x_train)
np.array(x_test)
np.array(x_val)

np.array(y_train)
np.array(y_test)
np.array(y_val)
"""
"""
dir = './Dataset_For_Model(20190927, All)'

label = ("감초", "당귀", "당삼", "대조", "도인", 
        "맥문동", "목향", "반하", "백작약", "백출",
        "백편두", "복령", "산약", "산조인", "숙지황", 
        "승마", "시호", "아교", "애엽", "오미자", "용안육",
        "원지", "육계", "인삼", "진피", "천궁", "하수오", "홍화", "황기")

(x_train, y_train), (x_test, y_test), (x_val, y_val) = Data_All_Preprocessing(dir=dir, label=label, X_size=224, Y_size=224)

print(y_train[0])
"""
"""
print(x_train[0])
print(y_train[0])

print(x_test[0])
print(y_test[0])

print(x_val[0])
print(y_val[0])
"""
