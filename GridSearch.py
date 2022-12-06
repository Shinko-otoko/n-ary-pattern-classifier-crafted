import cv2
import numpy as np
import math
import os
from os import listdir
from os.path import isfile, join
from math import *
import random
import time
import csv
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
from multiprocessing import Process
from multiprocessing import Manager
from joblib import dump, load
import ctypes
import numpy

from sklearn.model_selection import StratifiedKFold

fun = ctypes.CDLL("/home/pi/Desktop/Labo/FeaturesEngeeniring/Labo9/libfun.so")

def generate_angle_distance (angle_max, distance_max,nb_voisins): 
    angles=[]
    distances=[]
    for i in range (0,nb_voisins): 
        angles.append(random.randint(0,angle_max))
        distances.append(random.randint (0,distance_max))

    return angles, distances

def load_formule(chemin_formule):
    angle,distance=[],[]
    with open(chemin_formule, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            angle.append(row[0])
            distance.append(row[1])

    angle=np.array(angle).astype(np.float32)
    distance=np.array(distance).astype(np.float32)
    return [angle,distance]


def load_formules():
    liste_formules=[]
    onlyfiles = [f for f in listdir("formules/") if isfile(join("formules/", f))]
            
    for img_path in onlyfiles:
        liste_formules.append(load_formule("formules/"+img_path))


    return liste_formules


def LoadImgs_in_sub_directory(PathImgs):
        liste_noms=[]
        liste_subdirectory=[]
        liste_imgs=[]
        a=1
        onlydirectories = [f for f in listdir(PathImgs) if not isfile(join(PathImgs, f))]
        for directory in onlydirectories :

            onlyfiles = [f for f in listdir(PathImgs+directory+"/") if isfile(join(PathImgs+directory+"/", f))]
            #print (PathImgs+directory,onlyfiles )
            for img_path in onlyfiles:

                 # percent of original size



                    if a>0 :
                        img=cv2.imread(PathImgs+directory+"/"+img_path)
                        scale_percent = 50
                        width = int(img.shape[1] * scale_percent / 100)
                        height = int(img.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        img=cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
                        if( len(img)!=0):
                            liste_imgs.append(img)
                            liste_subdirectory.append(directory)
                        #print (PathImgs+directory+"/"+img_path,len(img), len(img[0]))
                            liste_noms.append(img_path)
                    a+=1



        return liste_noms,liste_subdirectory,liste_imgs


def LoadImgs_in_sub_sub_directory(PathImgs):
        liste_noms=[]
        liste_subdirectory=[]
        liste_imgs=[]
        a=1
        i=0
        onlydirectories = [f for f in listdir(PathImgs) if not isfile(join(PathImgs, f))]
        for directory in onlydirectories :

            onlydirectories2 = [f for f in listdir(PathImgs+directory+"/") if not isfile(join(PathImgs+directory+"/", f))]
            j=0
            for directory2 in onlydirectories2 :
                


                onlyfiles = [f for f in listdir(PathImgs+directory+"/"+directory2+"/") if isfile(join(PathImgs+directory+"/"+directory2+"/", f))]
            
                for img_path in onlyfiles:
                    
                    try :
                        #print (PathImgs+directory+"/"+directory2+"/"+img_path)
                 # percent of original size
                        if (not "c.png"  in img_path) and (not "a.png"  in img_path) :
                            #print (PathImgs+directory+"/"+directory2+"/"+img_path)
                            img=cv2.imread(PathImgs+directory+"/"+directory2+"/"+img_path)
                            scale_percent = 50
                            width = int(img.shape[1] * scale_percent / 100)
                            height = int(img.shape[0] * scale_percent / 100)
                            dim = (width, height)
                            img=cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
                            if( len(img)!=0):
                                liste_imgs.append(img)
                                liste_subdirectory.append(directory)
                        #print (PathImgs+directory+"/"+img_path,len(img), len(img[0]))
                                liste_noms.append(img_path)
                    except : 
                        e=0
                j+=1
                a+=1
            i+=1


        return liste_noms,liste_subdirectory,liste_imgs



def load_fleur_dataset():
    liste_nomsA,liste_subdirectoryA,liste_imgsA = LoadImgs_in_sub_directory("/home/pi/Desktop/Databases/Fleur/BBregroupAutre/")
    liste_nomsBBFlower,liste_labelsBBFlower,liste_imgsBBFlower = LoadImgs_in_sub_sub_directory("/home/pi/Desktop/Databases/Fleur/BBregroup/")
    for i in range (0,len(liste_imgsA )):
        liste_imgsBBFlower.append(liste_imgsA[i])
        liste_labelsBBFlower.append(liste_subdirectoryA[i])
        liste_nomsBBFlower.append(liste_nomsA[i])

    return liste_imgsBBFlower,liste_labelsBBFlower,liste_nomsBBFlower

liste_imgsBBFlower,liste_labelsBBFlower,liste_nomsBBFlower=load_fleur_dataset()






def compute_proba_c(probas,nb_pixels,nb_classes,len_X_train):
    probas=probas.astype(np.float32).reshape(-1)
    moyennes=np.zeros ((len_X_train,nb_classes)).astype(np.float32).reshape(-1)
    nb_pixels=np.int32(nb_pixels)
    nb_classes=np.int32(nb_classes)
    len_X_train=np.int32(len_X_train)
    probas_c=probas.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    moyennes_c=moyennes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    fun.myFunction2.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_int]
    fun.myFunction2(probas_c,moyennes_c, nb_pixels, len_X_train, nb_classes)

    array_length=len_X_train*nb_classes
    buffer = np.ctypeslib.as_array( (ctypes.c_float * array_length).from_address(ctypes.addressof(moyennes_c.contents)))
    buffer2=np.array(buffer)
    buffer=buffer2.reshape((len_X_train,nb_classes))

    return buffer


def compute_pixels_c(X_train,X_train_converted,angle,distance,nb_pixel): 
    dataset_train1=[]
    random.seed(0)

    NB_PIXEL_C=np.int32(nb_pixel)

    NUM=len(angle)

    fun.myFunction.argtypes = [ctypes.POINTER(ctypes.c_uint32),ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint32),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int32)]
 
    angle=np.array(angle).astype(np.int32)
    distance=np.array(distance).astype(np.int32)

    angles_c=angle.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    distances_c=distance.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    for j in range (0,len(X_train)):
        #print ("train_clfs l143 ",j,"/", len(X_train))


        img=X_train[j]

        #numpy.random.seed(0) 
        l_x=np.random.randint(len(img)-1, size=nb_pixel)
        l_y=np.random.randint(len(img[0])-1, size=nb_pixel)

        l_x_c=l_x.astype(np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        l_y_c=l_y.astype(np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))



        b_c=X_train_converted[j][0]
        g_c=X_train_converted[j][1]
        r_c=X_train_converted[j][2]

        

        # print (len(result));
        # exit(0)
        array_length=nb_pixel*((len(angle)+1)*3)

        result=np.zeros(array_length)
        result_c=result.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        width=np.int32(len(img))
        height=np.int32(len(img[0]))

        
    
        fun.myFunction(angles_c,NUM,distances_c,NUM, NB_PIXEL_C, b_c, g_c, r_c,l_x_c,l_y_c,height, width,result_c)     
 
        buffer = np.ctypeslib.as_array( (ctypes.c_int32 * array_length).from_address(ctypes.addressof(result_c.contents)))
        buffer2=np.array(buffer)
        # print (buffer)
        #print (200*((len(angle)+1)*3))
        buffer=buffer2.reshape((nb_pixel,((len(angle)+1)*3)))
        # print (buffer[0])
        # exit()
        for i in range (0,nb_pixel):
            dataset_train1.append(buffer[i])

    return dataset_train1



def convert_dataset(X_train ):
    dataset=[]
    
    for j in range (0,len(X_train)):
        img=X_train[j]
        b,g,r = cv2.split(img)

        b_c=b.reshape(-1).astype(np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        g_c=g.reshape(-1).astype(np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        r_c=r.reshape(-1).astype(np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))        
        dataset.append([b_c,g_c,r_c])

    return dataset



def train_clfs(clf1,clf2,X_train,y_train,X_train_converted,angle,distance,nb_pixel):
    dataset_train1=[]
    labels_train_1=[]
    random.seed(0)
    for j in range (0,len(X_train)):
        #print ("train_clfs l152 ",j, len(X_train))
        for i in range (0,nb_pixel):
            labels_train_1.append(y_train[j])

    dataset_train1=compute_pixels_c(X_train,X_train_converted,angle,distance,nb_pixel)
    #print (dataset_train1[0])
    clf1.fit(dataset_train1, labels_train_1)

    taille=len(clf1.predict_proba([dataset_train1[0]])[0])




    dataset_train2=[]
    labels_train2=[]

    buffer_probas=compute_pixels_c(X_train,X_train_converted,angle,distance,nb_pixel)
    proba=clf1.predict_proba(buffer_probas)
    compteur=0
    dataset_train2_c=compute_proba_c(proba,nb_pixel,taille,len(X_train))

    
    #scores = cross_val_score(clf2, dataset_train2, y_train, cv=5)

    clf2.fit(dataset_train2_c, y_train)

    return clf1,clf2






def build_test_dataset(clf1,clf2,X_test2,y_test,X_test_converted2,nb_pixel,angle,distance):
    taille=80
    random.seed(0)
    dataset_test=[]




    buffer_probas=compute_pixels_c(X_test2,X_test_converted2,angle,distance,nb_pixel)
    proba=clf1.predict_proba(buffer_probas)
    compteur=0

    dataset_test_c=compute_proba_c(proba,nb_pixel,taille,len(X_test2))



    return dataset_test_c
    #dataset_final.append(np.histogram(matrix_tmp.reshape(-1), bins=10)[0])
    #print (np.histogram(matrix_tmp.reshape(-1), bins=10)[0])

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def save_data_set(filename,dataset,labels,noms):
    print ("save_data_set")
    with open(filename, mode='w',newline='') as fichier:
        writer = csv.writer(fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        tmp=[]
        for j in range (0,len(dataset[0])):
            tmp.append("F"+str(j))
        tmp.append("classe")
        tmp.append("nom")
        writer.writerow(tmp)
        
        for j in range (0,len(dataset)):
            tmp=[]
            for m in range (0, len(dataset[j])):
                tmp.append(round_up(float(dataset[j][m])))
            tmp.append("classe"+str(labels[j]))
            tmp.append(noms[j])
            writer.writerow(tmp)

def concatenate_dataset(dataset_app,dataset_test,dataset_app_tmp,dataset_test_tmp): 
    dataset2_app=[]
    dataset2_test=[]
    for i in range (0,len(dataset_app)): 
        dataset2_app.append(np.concatenate((dataset_app[i],dataset_app_tmp[i])))

    for i in range (0,len(dataset_test)): 
        dataset2_test.append(np.concatenate((dataset_test[i],dataset_test_tmp[i])))

    return dataset2_app,dataset2_test

def train_and_evaluate (X_train, X_train_converted, y_train, X_test,X_test_converted, y_test,n1,n2,n3,n4,n5):


    clf1 = RandomForestClassifier(n_estimators=n5,random_state=0,max_depth=n4,n_jobs=-1,min_samples_split=30)
    clf2 = RandomForestClassifier(n_estimators=100,random_state=0,max_depth=20,n_jobs=-1)

    dataset_app=[]
    dataset_test=[]
    for i in range (0,n3):
        angles, distances=generate_angle_distance (360, 200,n2)  
        nb_pixels=n1  
        train_clfs(clf1,clf2,X_train,y_train,X_train_converted,angles,distances,nb_pixels)
        dataset_app_tmp=build_test_dataset(clf1,clf2,X_train,y_train,X_train_converted,nb_pixels,angles,distances)    
        dataset_test_tmp=build_test_dataset(clf1,clf2,X_test,y_test,X_test_converted,nb_pixels,angles,distances)
        if (i==0): 
            dataset_app,dataset_test=dataset_app_tmp,dataset_test_tmp
        else:
            dataset_app,dataset_test=concatenate_dataset(dataset_app,dataset_test,dataset_app_tmp,dataset_test_tmp)

    clf = RandomForestClassifier(n_estimators=100,random_state=0,max_depth=20)
    clf.fit(dataset_app_tmp, y_train)
    y_pred=clf.predict(dataset_test_tmp)
    return accuracy_score(y_test, y_pred),dataset_app,dataset_test


                        
def save_l_datasets(l_dataset_app,l_y_train,l_dataset_test,l_y_test):
    pid = os.getpid()

    try:
        os.mkdir("datasets")
    except OSError:
        a=0
    else:
        a=0


    try:
        os.mkdir("datasets/"+str(pid))
    except OSError:
        a=0
    else:
        a=0

    for i in range (0,len(l_dataset_app)):
        save_data_set("datasets/"+str(pid)+"/"+str(i)+"_app.csv",l_dataset_app[i],l_y_train[i],l_y_train[i])
        save_data_set("datasets/"+str(pid)+"/"+str(i)+"_test.csv",l_dataset_test[i],l_y_test[i],l_y_test[i])

def grid_search (N1,N2,N3,N4,N5):
    len_search_space= len(N1)*len(N2)*len(N3)*len(N4)*len(N5)
    skf = StratifiedKFold(n_splits=3)
    X,y,noms=load_fleur_dataset()


    X=np.array(X)
    y=np.array(y)






    max_score=0
    max_score_parameters=[]
    max_score_l_dataset_app=[]
    max_score_l_dataset_test=[]
    max_score_l_y_train=[]
    max_score_l_y_test=[]

    compteur=0
    for n1 in N1:
        for n2 in N2: 
            for n3 in N3: 
                for n4 in N4 : 
                    for n5 in N5:
                        compteur+=1
                        scores=[]
                        
                        l_dataset_app=[]
                        l_dataset_test=[]
                        l_y_train=[]
                        l_y_test=[]
                        for train_index, test_index in skf.split(X, y):
                            X_train, X_test = X[train_index], X[test_index]
                            X_train_converted, X_test_converted = convert_dataset(X_train), convert_dataset(X_test)
                            y_train, y_test = y[train_index], y[test_index]
                            random.seed(0)
                            numpy.random.seed(0)   
                            score,dataset_app,dataset_test=train_and_evaluate (X_train, X_train_converted, y_train, X_test,X_test_converted, y_test,n1,n2,n3,n4,n5)
                            l_dataset_app.append(dataset_app)
                            l_dataset_test.append(dataset_test)
                            l_y_train.append(y_train)
                            l_y_test.append(y_test)
                            scores.append(score)
                        scores=np.array(scores)
                        if (np.mean(scores)>max_score):
                            max_score=np.mean(scores)
                            max_score_parameters=[n1,n2,n3,n4,n5]
                            max_score_l_dataset_app=l_dataset_app
                            max_score_l_dataset_test=l_dataset_test
                            max_score_l_y_train=l_y_train
                            max_score_l_y_test=l_y_test
                        print("pos in search space:", compteur,"/",len_search_space) 
                        print ("n1:",n1, "n2:", n2, "n3:",n3, "n4:",n4,"n5:",n5, "tx identification:",np.mean(scores),"max_score:",max_score)

    save_l_datasets(max_score_l_dataset_app,max_score_l_y_train,max_score_l_dataset_test,max_score_l_y_test)

                        
    print ("Le score max est : ",max_score, " les paramètres :",max_score_parameters)

def main(): 
    start=time.time()
    pid = os.getpid()
    nb_pixel=250
    N1=[400,500]
    N2=[5,10,15,20]
    N3=[2,3,4,5,6]
    N4=[30]
    N5=[30]
    grid_search (N1,N2,N3,N4,N5)
    end=time.time()
    print("Total time elapsed: ",end-start)





if __name__ == '__main__':
    main()




