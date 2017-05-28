import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


def get_train_test():
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
