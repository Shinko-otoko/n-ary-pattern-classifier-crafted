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



fun = ctypes.CDLL("/home/pi/Desktop/Labo/FeaturesEngeeniring/Labo10/libfun.so")


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
        print ("train_clfs l143 ",j,"/", len(X_train))


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
        print ("train_clfs l152 ",j, len(X_train))
        for i in range (0,nb_pixel):
            labels_train_1.append(y_train[j])

    dataset_train1=compute_pixels_c(X_train,X_train_converted,angle,distance,nb_pixel)
    print (dataset_train1[0])
    clf1.fit(dataset_train1, labels_train_1)

    taille=len(clf1.predict_proba([dataset_train1[0]])[0])




    dataset_train2=[]
    labels_train2=[]

    buffer_probas=compute_pixels_c(X_train,X_train_converted,angle,distance,nb_pixel)
    proba=clf1.predict_proba(buffer_probas)
    compteur=0

    for j in range (0,len(X_train)):
        moyenne=np.zeros(taille)
        for i in range (0,nb_pixel):
            for mp in range (0,taille-1):
                moyenne[mp]+=proba[compteur][mp]
            compteur+=1
        dataset_train2.append(moyenne)
    
    scores = cross_val_score(clf2, dataset_train2, y_train, cv=5)

    clf2.fit(dataset_train2, y_train)

    return clf1,clf2,scores.mean()





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





def train_clfs_c(clf1,clf2,X_train,y_train,X_train_converted,angle,distance,nb_pixel):
    dataset_train1=[]
    labels_train_1=[]
    random.seed(0)
    for j in range (0,len(X_train)):
        print ("train_clfs l152 ",j, len(X_train))
        for i in range (0,nb_pixel):
            labels_train_1.append(y_train[j])

    dataset_train1=compute_pixels_c(X_train,X_train_converted,angle,distance,nb_pixel)
    print (dataset_train1[0])
    clf1.fit(dataset_train1, labels_train_1)

    taille=len(clf1.predict_proba([dataset_train1[0]])[0])




    dataset_train2=[]
    labels_train2=[]

    buffer_probas=compute_pixels_c(X_train,X_train_converted,angle,distance,nb_pixel)
    proba=clf1.predict_proba(buffer_probas)
    compteur=0

    start1=time.time()
    for j in range (0,len(X_train)):
        moyenne=np.zeros(taille)
        for i in range (0,nb_pixel):
            for mp in range (0,taille-1):
                moyenne[mp]+=proba[compteur][mp]
            compteur+=1
        dataset_train2.append(moyenne)
    end1=time.time()

    start2=time.time()
    dataset_train2_c=compute_proba_c(proba,nb_pixel,taille,len(X_train))
    end2=time.time()

    print ("et1:",end1-start1,"et2:",end2-start2)

    print (dataset_train2[1])
    print (dataset_train2_c[1])
    exit()





    scores = cross_val_score(clf2, dataset_train2, y_train, cv=5)

    clf2.fit(dataset_train2, y_train)

    return clf1,clf2,scores.mean()





liste_formules=load_formules()

nb_pixel=25

X_train, X_test, y_train, y_test = train_test_split( liste_imgsBBFlower, liste_labelsBBFlower, test_size=0.33, random_state=42)



def build_test_dataset(clf1,clf2,X_test2,y_test,X_test_converted2,nb_pixel,angle,distance):
    taille=80
    random.seed(0)
    dataset_test=[]




    buffer_probas=compute_pixels_c(X_test2,X_test_converted2,angle,distance,nb_pixel)
    proba=clf1.predict_proba(buffer_probas)
    compteur=0

    for j in range (0,len(X_test2)):
        moyenne=np.zeros(taille)
        for i in range (0,nb_pixel):
            for mp in range (0,taille-1):
                moyenne[mp]+=proba[compteur][mp]
            compteur+=1
        dataset_test.append(moyenne)



    return dataset_test
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

def thread_function(liste_clf1,liste_clf2,X_train,y_train,X_train_converted,X_test_converted,angle,distance,l_dataset_app,l_dataset_test,pid, id_thread,nb_pixel):
    print ("debut thread : ",id_thread)

    random.seed(0)
    
    
    clf1 = RandomForestClassifier(n_estimators=5,random_state=0,max_depth=5,n_jobs=-1)
    clf2 = RandomForestClassifier(n_estimators=10,random_state=0,max_depth=10,n_jobs=-1)
    train_clfs_c(clf1,clf2,X_train,y_train,X_train_converted,angle,distance,nb_pixel)


    
    #dump(clf1, "output/output_process_"+str(pid)+"/"+"thread"+str(id_thread)+'_clf1.joblib') 
    #dump(clf2, "output/output_process_"+str(pid)+"/"+"thread"+str(id_thread)+'_clf2.joblib')

    dataset_app_tmp=build_test_dataset(clf1,clf2,X_train,y_train,X_train_converted,nb_pixel,angle,distance)    

    save_data_set("output/output_process_"+str(pid)+"/"+"thread"+str(id_thread)+'_ww_dataset_app.joblib',dataset_app_tmp,y_train,y_train)

    dataset_test_tmp=build_test_dataset(clf1,clf2,X_test,y_test,X_test_converted,nb_pixel,angle,distance)
    
    save_data_set("output/output_process_"+str(pid)+"/"+"thread"+str(id_thread)+'_ww_dataset_test.joblib',dataset_test_tmp,y_test,y_test)



    # 
    # l_dataset_app.append(dataset_app_tmp)
    # 
    # l_dataset_test.append(dataset_test_tmp)

    print ("fin thread : ",id_thread)




if __name__ == '__main__':
    pid = os.getpid()

    nb_pixel=25


    try:
        os.mkdir("output/output_process_"+str(pid))
    except OSError:
        print ("Creation of the directory  failed" )
    else:
        print ("Successfully created the directory  " )



    manager = Manager()
#nb_thread=len(liste_formules)
    nb_thread=1
    liste_clf1=[]
    liste_clf2=[]

    l_dataset_app = manager.list()
    l_dataset_test = manager.list()

    liste_angle,liste_distance,liste_score,liste_thread=[],[],[],[]
    liste_clf1, liste_clf2=[],[]




    for id_thread in range (0,nb_thread):
            liste_score.append(0.)
            angle,distance=liste_formules[id_thread]
            liste_clf1.append(tree.DecisionTreeClassifier(random_state=0))
            liste_clf2.append(RandomForestClassifier(n_estimators=20,random_state=0,max_depth=10))

            X_train_converted=convert_dataset(X_train )
            X_test_converted=convert_dataset(X_test )
            #try:
            thread_function(liste_clf1,liste_clf2,X_train,y_train,X_train_converted,X_test_converted,angle,distance,l_dataset_app,l_dataset_test, pid,  id_thread,nb_pixel ) 
        
            #t=threading.Thread( target=thread_function, args=(liste_clf1,liste_clf2,X_train,y_train,angle,distance,l_dataset_app,l_dataset_test, id_thread, ) )

            nb_pixel+=50
            #except:
            #    print ("Error: unable to start thread-1")

    for t in liste_thread:
            t.join()

    print ("finitosh")








# # #scores = cross_val_score(clf, dataset_app, y_train, cv=5)



# for mop in range (2,len(liste_formules)):
#     random.seed(0)

#     clf1 = RandomForestClassifier(n_estimators=20,random_state=0)
#     clf2 = RandomForestClassifier(n_estimators=20,random_state=0)
#     print (mop,"/",len(liste_formules))

#     for i in range (0,mop):
#         clf1 = RandomForestClassifier(n_estimators=20,random_state=0)
#         clf2 = RandomForestClassifier(n_estimators=20,random_state=0)
 
#         angle,distance=liste_formules[i]
#         train_clfs(clf1,clf2,X_train,y_train,angle,distance,nb_pixel)

#         dataset_app_tmp=build_test_dataset(clf1,clf2,X_train,y_train,nb_pixel,angle,distance)     
#         l_dataset_app.append(dataset_app_tmp)
#         dataset_test_tmp=build_test_dataset(clf1,clf2,X_test,y_test,nb_pixel,angle,distance)
#         l_dataset_test.append(dataset_test_tmp)


       


#     dataset_app=[]
#     dataset_test=[]


    

#     for i in range (0,len(l_dataset_app[0])):
#         dataset_app.append(np.array([]))

#         for j in range (1,mop):
#             dataset_app[i]=np.concatenate((dataset_app[i],l_dataset_app[j][i]))


#     print (len(dataset_app),len(dataset_app[0]))

#     for i in range (0,len(l_dataset_test[0])):
#         dataset_test.append(np.array([]))

#         for j in range (1,mop):
#             dataset_test[i]=np.concatenate((dataset_test[i],l_dataset_test[j][i]))


# # print (len(dataset_app[0]),len(dataset_test[0]))
    
#     clf = RandomForestClassifier(n_estimators=20,random_state=0)
#     clf.fit(dataset_app, y_train)

#     y_pred=clf.predict(dataset_test)
#     print ("résultat :",accuracy_score(y_test, y_pred))
# # #scores = cross_val_score(clf, dataset_app, y_train, cv=5)

# # print (scores.mean())
