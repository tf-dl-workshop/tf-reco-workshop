import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


def get_train_test():
    if os.path.exists('data/_numpydata/train_arr.npy') & os.path.exists('data/_numpydata/test_arr.npy'):
        return np.load('data/_numpydata/train_arr.npy'), np.load('data/_numpydata/test_arr.npy')
    else:
        cols = ['user_id', 'movie_id', 'rating', 'timestamp']

        rating_df = pd.read_csv('data/ratings.dat', sep='::', names=cols)

        train, test = train_test_split(rating_df, test_size=0.1)

        item_index = dict()
        user_index = dict()

        for idx, id in enumerate(train.movie_id.drop_duplicates()):
            item_index[id] = idx
        for idx, id in enumerate(train.user_id.drop_duplicates()):
            user_index[id] = idx

        train_arr = np.zeros(shape=[len(item_index), len(user_index)])
        test_arr = np.zeros(shape=[len(item_index), len(user_index)])

        for _, rating in train.iterrows():
            train_arr[item_index[rating.movie_id], user_index[rating.user_id]] = rating.rating

        for _, rating in test.iterrows():
            if (rating.movie_id in item_index) & (rating.user_id in user_index):
                test_arr[item_index[rating.movie_id], user_index[rating.user_id]] = rating.rating
            else:
                pass

        np.save('data/_numpydata/train_arr.npy', train_arr)
        np.save('data/_numpydata/test_arr.npy', test_arr)

        return train_arr, test_arr
