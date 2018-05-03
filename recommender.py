import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
import random
import h5py as h
from scipy.sparse.linalg import svds
import urllib.request
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import operator
import math


class Recommender:
    def __init__(self):
        f = h.File('dataset/tags_U.hdf5', 'r')
        self.U = f['tags_U'][:]
        f.close()
        #print(self.V.shape)
        f = h.File('dataset/book_indices_val.hdf5', 'r')
        self.book_indices_val = f['book_indices_val'][:]
        self.book_indices = {x[1]:x[0] for x in enumerate(self.book_indices_val)}
        f.close()
        self.K = 20
        
    # get similarity to the most similar book from the book set
    def getBookSimilarity(self, book_id, book_set):
        tags_matrix = self.U
        row = self.book_indices[book_id]
        v1 = tags_matrix[row,:]
        v1 = v1.reshape(1,-1)
        similarities = []
        for b in book_set:
            row = self.book_indices[b]
            v2 = tags_matrix[row,:]
            v2 = v2.reshape(1,-1) 
            sim = cosine_similarity(v1, v2)
            similarities.append(sim)
        max_sim = max(similarities)
        return max_sim[0][0]

    # perform factorization
    def matrixFactorization(self, K=20, steps=100, alpha=0.005, beta=0.02):
        f = h.File('dataset/ratings_matrix.hdf5', 'r')
        R = f['ratings_matrix'][:]
        f.close()
            
        self.K = K
        print("Performing factorization")
        n = R.shape[0]
        m = R.shape[1]
        P = np.random.rand(n, K)
        Q = np.random.rand(m, K)
        Q = Q.T
        for step in range(steps):
            print("step " + str(step))
            for i in range(R.shape[0]):
                #print(i)
                for j in range(R.shape[1]):
                    #print(R[i,j])
                    if R[i,j] > 0:
                        eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                        for k in range(K):
                            P[i,k] = P[i,k] + alpha * (eij * Q[k,j] - beta * P[i,k])
                            Q[k,j] = Q[k,j] + alpha * (eij * P[i,k] - beta * Q[k,j])
            
##            e = 0
##            for i in range(R.shape[0]):
##                for j in range(R.shape[1]):
##                    if R[i,j] > 0:
##                        e = e + pow(R[i,j] - np.dot(P[i,:], Q[:,j]), 2)
##                        for k in range(K):
##                            e = e + (beta/2) * (pow(P[i,k], 2) + pow(Q[k,j], 2))
##            if e < 0.001:
##                break
        f = h.File('dataset/P_matrix.hdf5', 'w')
        dset = f.create_dataset('P_matrix', data=P)
        f.close()
        f = h.File('dataset/Q_matrix.hdf5', 'w')
        dset = f.create_dataset('Q_matrix', data=Q)
        f.close()
        print("done")
##        eR = np.dot(P,Q)
##        f = h.File('dataset/eR.hdf5', 'w')
##        dset = f.create_dataset('eR', data=eR)
##        f.close()

    # add a new user to the model
    def addUser(self, user_ratings, steps=50, alpha=0.005, beta=0.02):
        f = h.File('dataset/P_matrix.hdf5', 'r')
        P = f['P_matrix'][:]
        f.close()
        f = h.File('dataset/Q_matrix.hdf5', 'r')
        Q = f['Q_matrix'][:]
        f.close()

        user_vector = np.zeros(len(self.book_indices_val))
        for b, r in user_ratings:
            user_vector[np.where(self.book_indices_val == b)] = r

        Pu = np.random.rand(1, self.K)

        P = np.vstack([P, Pu])
        u_ind = P.shape[0]-1
        #print(R.shape)
        #print(P.shape)
        for step in range(steps):
            for j in range(len(user_vector)):
                if user_vector[j] > 0:
                    eij = user_vector[j] - np.dot(P[u_ind,:], Q[:,j])
                    for k in range(self.K):
                        P[u_ind,k] = P[u_ind,k] + alpha * (eij * Q[k,j] - beta * P[u_ind,k])

        read_books = [x[0] for x in user_ratings]
        est_ratings = np.zeros(len(user_vector))
        # select books the user liked:
        liked_books = [x[0] for x in user_ratings if x[1] >= 4]
        for i in range(len(user_vector)):
            est_ratings[i] = np.dot(P[u_ind,:], Q[:,i])
            if len(liked_books) != 0:
                est_ratings[i] = est_ratings[i] - (0.5 - self.getBookSimilarity(self.book_indices_val[i], liked_books))/5

        top_ind = sorted(range(len(est_ratings)), key=lambda i: est_ratings[i])
        top_ind = list(reversed(top_ind))
        rec_books = []
        for i in top_ind:
            if (self.book_indices_val[i] not in read_books):
                rec_books.append(self.book_indices_val[i])

        return rec_books[:160]

    def getRecommendations(self, user_ratings, mode='MF'):

        if mode == 'MF':
            rec_books = self.addUser(user_ratings)
            return rec_books


