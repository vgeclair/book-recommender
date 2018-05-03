import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
import random
import h5py as h
from scipy.sparse.linalg import svds
import urllib.request
from scipy import stats
import os
import time
from sklearn import preprocessing
import math
import operator


# get tags matrix and perform SVD
def get_tags_matrix():
    tags = pd.read_csv('dataset/book_tags.csv')
    print("Building tags matrix")

    threshold = 1
    tags = tags[tags['count'] > threshold]

    tag_ids = tags['tag_id'].unique()
    book_ids = tags['goodreads_book_id'].unique()
    row_num = len(book_ids)
    col_num = len(tag_ids) 
    tags_matrix = lil_matrix((row_num, col_num), dtype='d')
    tag_indices = {x[1]:x[0] for x in enumerate(tag_ids)}
    bookt_indices = {x[1]:x[0] for x in enumerate(book_ids)}

    for index, row in tags.iterrows():
        tg = row['tag_id']
        bk = row['goodreads_book_id']
        cnt = row['count']
        tags_matrix[bookt_indices[bk], tag_indices[tg]] = int(cnt)

    tags_matrix = preprocessing.normalize(tags_matrix, norm='l2')

##    f = h.File('dataset/tags_matrix.hdf5', 'w')
##    dset = f.create_dataset('tags_matrix', data=tags_matrix.todense())
##    f.close()

    U, S, Vt = svds(tags_matrix, k=100)

    f = h.File('dataset/tags_U.hdf5', 'w')
    dset = f.create_dataset('tags_U', data=U)
    f.close()
    f = h.File('dataset/tags_Vt.hdf5', 'w')
    dset = f.create_dataset('tags_Vt', data=Vt)
    f.close()
    print("done")


def get_ratings_matrix():
    ratings = pd.read_csv('dataset/ratings.csv')

    print("Cleaning the dataset")
    #remove duplicate ratings
    ratings = ratings.drop_duplicates(subset=['user_id', 'book_id'], keep="last")

##    #save the frequency for each book
##    books = pd.read_csv('dataset/books.csv')
##    books = pd.DataFrame(books, columns=['id', 'ratings_count'])
##    books.to_csv('dataset/frequencies.csv', index=False)

    ratings = pd.DataFrame(ratings, columns=['book_id', 'user_id', 'rating'])

    #remove users that rated less than 10 books
    threshold = 10
    value_counts = ratings['user_id'].value_counts()
    to_remove = value_counts[value_counts <= threshold].index
    ratings = ratings.loc[~ratings['user_id'].isin(to_remove),:]

    #remove 50% of users
    to_keep = ratings['user_id'].unique().tolist()
    to_keep = random.sample(to_keep, int(len(to_keep)/2))
    ratings = ratings[ratings['user_id'].isin(to_keep)]

    print("done")
    print("Building the ratings matrix")

    # build the ratings matrix
    users = ratings['user_id'].unique()
    books = ratings['book_id'].unique()
    print(books)
    user_indices = {x[1]:x[0] for x in enumerate(users)}
    book_indices = {x[1]:x[0] for x in enumerate(books)}

    f = h.File('dataset/user_indices_val.hdf5', 'w')
    dset = f.create_dataset('user_indices_val', data=users)
    f.close()
    f = h.File('dataset/book_indices_val.hdf5', 'w')
    dset = f.create_dataset('book_indices_val', data=books)
    f.close()

    row_num = len(users)
    col_num = len(books) 
    ratings_matrix = np.zeros((row_num, col_num), dtype=int)

    for index, row in ratings.iterrows():
        usr = row['user_id']
        bk = row['book_id']
        rt = row['rating']
        ratings_matrix[user_indices[usr], book_indices[bk]] = int(rt)

    #print(ratings_matrix)

    f = h.File('dataset/ratings_matrix.hdf5', 'w')
    dset = f.create_dataset('ratings_matrix', data=ratings_matrix)
    f.close()

    print("done")
    print("Computing entropies")

   #compute entropies
    vals = ratings.groupby('book_id')['rating'].apply(list)
    ent = []
    book_id = vals.index
    for r in vals:
        ent.append(float(stats.entropy(r)))
    d = {'book_id': book_id, 'entropy': ent}
    df = pd.DataFrame(data=d)
    df.to_csv('dataset/entropies.csv', index=False)
    
    print("Computing coverage")
    #compute coverage
    coverage = []
    ids = books
    for col in range(ratings_matrix.shape[1]):
        cv = 0
        for row in range(ratings_matrix.shape[0]):
            if (ratings_matrix[row, col] != 0):
                for col2 in range(ratings_matrix.shape[1]):
                    if (ratings_matrix[row, col2] != 0) and (col2 != col):
                        cv = cv +1
        coverage.append(cv)

    d = {'book_id': ids, 'coverage': coverage}
    df = pd.DataFrame(data=d)
    df.to_csv('dataset/coverage.csv', index=False)

    print("Computing popularity")
    # compute popularity
    popularity = []
    ids = books
    for col in range(ratings_matrix.shape[1]):
        p = 0
        for row in range(ratings_matrix.shape[0]):
            if (ratings_matrix[row, col] != 0):
                p = p + 1
        popularity.append(p)

    d = {'book_id': ids, 'popularity': popularity}
    df = pd.DataFrame(data=d)
    df.to_csv('dataset/popularity.csv', index=False)

    print("done")

    #f = h.File('dataset/ratings_matrix.hdf5', 'r')
    #matrix = f['ratings_matrix'][:]
    #f.close()

# download book covers
def load_images():
    books = pd.read_csv('dataset/books.csv')
    for index, row in books.iterrows():
        url = row['image_url']
        name = 'img/' + str(row['id']) + '.jpg'
        if os.path.exists(name):
            continue
        print(url)
        urllib.request.urlretrieve(url, name)

# sort values and save them as vectors
def get_AL():

    print("Preparing lists for AL")

##    popularity = pd.read_csv('dataset/popularity.csv')
##    popularity = popularity.sort_values(by='popularity', ascending=False)

    books = pd.read_csv('dataset/books.csv')
    books = pd.DataFrame(books, columns=['id', 'ratings_count'])
    best = books.sort_values(by=['ratings_count'],ascending=False)
    ids = best['id'].tolist()
    f = h.File('dataset/AL_books.hdf5', 'w')
    dset = f.create_dataset('AL_popularity', data=ids)
    f.close()

    entropy = pd.read_csv('dataset/entropies.csv')
    values = []
    for index, row in books.iterrows():
        r = entropy.loc[entropy['book_id'] == row['id']]
        r = next(r.iterrows())[1]
        e = r['entropy']
        p_entr = math.log(row['ratings_count']) * e
        values.append((row['id'], p_entr))
        
    values.sort(key=operator.itemgetter(1), reverse=True)

    entropy = [x[0] for x in values]
    #entropy = entropy.sort_values(by='entropy', ascending=False)
    f = h.File('dataset/AL_entropy.hdf5', 'w')
    #dset = f.create_dataset('AL_books', data=entropy['book_id'].values)
    dset = f.create_dataset('AL_books', data=entropy)
    f.close()

    coverage = pd.read_csv('dataset/coverage.csv')
    coverage = coverage.sort_values(by='coverage', ascending=False)
    f = h.File('dataset/AL_coverage.hdf5', 'w')
    dset = f.create_dataset('AL_books', data=coverage['book_id'].values)
    f.close()

    print("done")

#get_ratings_matrix()

#get_tags_matrix()

#load_images()

#get_AL()
