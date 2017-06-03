import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


def ae_train_test():
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']

    rating_df = pd.read_csv('data/ml-100k/u.data', sep='\t', names=cols)

    seed = 123
    while True:
        train, test = train_test_split(rating_df, test_size=0.2, random_state=seed)
        train_matrix = train.pivot(index='movie_id', columns='user_id', values='rating').fillna(0.0)
        test_matrix = test.pivot(index='movie_id', columns='user_id', values='rating').fillna(0.0)
        if len(set(train_matrix.columns).difference(set(test_matrix.columns))) > 0:
            seed += 1
        else:
            break

    test_matrix = test_matrix.filter(items=train_matrix.index, axis=0)

    train_eval_matrix = train_matrix.loc[test_matrix.index, test_matrix.columns]

    return train_matrix.as_matrix(), test_matrix.as_matrix(), train_eval_matrix.as_matrix()


def mf_train_test():
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    rating_df = pd.read_csv('data/ml-100k/u.data', sep='\t', names=cols)
    seed = 123
    train, test = train_test_split(rating_df, test_size=0.2, random_state=seed)
    train.is_copy = False
    test.is_copy = False
    test = test[(test.movie_id.isin(train.movie_id)) & (test.user_id.isin(train.user_id))]
    train.loc[:, 'user_id'] += -1
    test.loc[:, 'user_id'] += -1
    train.loc[:, 'movie_id'] += -1
    test.loc[:, 'movie_id'] += -1

    train_dict = dict()
    test_dict = dict()

    train_dict['user_id'] = train.user_id.as_matrix()
    train_dict['movie_id'] = train.movie_id.as_matrix()
    train_dict['rating'] = train.rating.as_matrix().astype(np.float32)

    test_dict['user_id'] = test.user_id.as_matrix()
    test_dict['movie_id'] = test.movie_id.as_matrix()
    test_dict['rating'] = test.rating.as_matrix().astype(np.float32)

    return train_dict, test_dict
