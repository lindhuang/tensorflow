#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:03:12 2019

@author: lindahuang
"""
import numpy as np
import matplotlib.pyplot as plt
file=open("results.txt")
file2=open("results2.txt")
file3=open("results3.txt")
file4=open("results4.txt")
file5=open("results5.txt")
file6=open("results6.txt")
accuracies=[]
accuracies2=[]
accuracies3=[]
accuracies4=[]
accuracies5=[]
accuracies6=[]

#for line in file.readlines():
#    accuracies.append(line)
#for line in file2.readlines():
#    accuracies2.append(line)
#for line in file3.readlines():
#    accuracies3.append(float(line))
#for line in file4.readlines():
#    accuracies4.append(float(line))
for line in file5.readlines():
    accuracies5.append(float(line))
for line in file6.readlines():
    accuracies6.append(float(line))
    
NumberRange1 = np.logspace(np.log10(1e-3), np.log10(1e-5), num=5)
NumberRange2 = np.logspace(np.log10(100),np.log10(10000),num=5)
NumberRange3 = np.logspace(np.log10(100),np.log10(500),num=10)
NumberRange4 = np.logspace(np.log10(0.2),np.log10(0.8),num=10)
NumberRange5 = np.logspace(np.log10(0.01),np.log10(1),num=5)
NumberRange6 = np.logspace(np.log10(0.2),np.log10(1),num=5)
    
#plt.plot(NumberRange2,accuracies2)
#plt.title('Hyperparameter:iterations')
#plt.xlabel('iterations')
#plt.ylabel('Accuracy')
#plt.show()

#plt.plot(NumberRange3,accuracies3)
#plt.title('Hyperparameter:batch size')
#plt.xlabel('batch size')
#plt.ylabel('Accuracy')
#plt.show()

#plt.plot(NumberRange4,accuracies4)
#plt.title('Hyperparameter:dropout')
#plt.xlabel('dropout')
#plt.ylabel('Accuracy')
#plt.show()

plt.plot(NumberRange5,accuracies5)
plt.title('Hyperparameter:stdev')
plt.xlabel('stdev')
plt.ylabel('Accuracy')
plt.show()

plt.plot(NumberRange6,accuracies6)
plt.title('Hyperparameter:keep prob')
plt.xlabel('keep prob')
plt.ylabel('Accuracy')
plt.show()
