#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from Analisys import *
import sys

rng = np.random.RandomState(22)

dimensions = []

dimensions.append(["512", "512"])     #1MB
dimensions.append(["512", "1024"])     #2MB
dimensions.append(["512", "2048"])     #4MB
dimensions.append(["512", "4096"])    #8MB
dimensions.append(["512", "8192"])    #16MB
dimensions.append(["512", "16384"])    #32MB
dimensions.append(["512", "32768"])    #64MB
dimensions.append(["512", "65536"])   #128MB
dimensions.append(["512", "131072"])   #256MB
dimensions.append(["512", "262144"])   #512MB
dimensions.append(["512", "524288"])  #1GB
dimensions.append(["512", "1048576"])  #2GB
dimensions.append(["512", "2097152"])  #4GB

'''
dimensions.append(["512", "1048576"]) #8GB
dimensions.append(["512", "2097152"]) #16GB
dimensions.append(["512", "4194304"]) #32GB
dimensions.append(["512", "8388608"]) #64GB
dimensions.append(["512", "16777216"]) #128GB
dimensions.append(["512", "33554432"]) #256GB

dimensions.append(["512", "67108864"]) #512GB
dimensions.append(["512", "134217728"]) #1TB
dimensions.append(["512", "268435456"]) #2TB
dimensions.append(["512", "536870912"]) #4TB
dimensions.append(["512", "1073741824"]) #8TB'''

#Multiplicação de matrizes
def dot(lines, columns, vezes, shared):
    for i in range(0, vezes):

        m1 = getMatrix(lines, columns)
        m2 = getMatrix(columns, lines)

        if shared:
            dot_matrix_share(m1, m2)
        else:
            dot_matrix(m1, m2)

#Hadamard
def mul(lines, columns, vezes, shared):
    for i in range(0, vezes):

        m1 = getMatrix(lines, columns)
        m2 = getMatrix(lines, columns)

        if shared:
            multiply_matrix_share(m1, m2)
        else:
            multiply_matrix(m1, m2)

#Add
def add(lines, columns, vezes, shared):
    for i in range(0, vezes):

        m1 = getMatrix(lines, columns)
        m2 = getMatrix(lines, columns)

        if shared:
            add_matrix_share(m1, m2)
        else:
            add_matrix(m1, m2)

#Sigmoid
def my_sigmoid(lines, columns, vezes, shared):
    for i in range(0, vezes):

        m1 = getMatrix(lines, columns)

        if shared:
            sigmoid_share(m1)
        else:
            sigmoid(m1)

#SoftMax
def soft_max(lines, columns, vezes, shared):
    for i in range(0, vezes):

        m1 = getMatrix(lines, columns)

        if shared:
            softMax_share(m1)
        else:
            softMax(m1)

#Scalar
def scalarOp(lines, columns, vezes, shared):
    for i in range(0, vezes):

        m1 = getMatrix(lines, columns)

        if shared:
            scalar_matrix_share(2, m1)
        else:
            scalar_matrix(2, m1)

#Tanh
def my_tanh(lines, columns, vezes, shared):
    for i in range(0, vezes):

        m1 = getMatrix(lines, columns)

        if shared:
            tanh_share(m1)
        else:
            tanh(m1)

def getMatrix(lines, columns):
    return np.asmatrix(rng.rand(lines, columns), config.floatX)

if __name__ == "__main__":
    lista = sys.argv

    operation = int(lista[1])
    lines = int(lista[2])
    columns = int(lista[3])
    times = 1#int(lista[4])
    shared = bool(int(lista[5]))

    if operation == 0:
        dot(lines, columns, times, shared)
    elif operation == 1:
        mul(lines, columns, times, shared)
    elif operation == 2:
        add(lines, columns, times, shared)
    elif operation == 3:
        my_sigmoid(lines, columns, times, shared)
    elif operation == 4:
        soft_max(lines, columns, times, shared)
    elif operation == 5:
        scalarOp(lines, columns, times, shared)
    elif operation == 6:
        my_tanh(lines, columns, times, shared)
