#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE


def learn_embedding(df, dims=2):
    X = df._get_numeric_data().values

    X_embedded = TSNE(n_components=dims).fit_transform(X)
    X_embedded.shape
    return X_embedded

if __name__ == "__main__":
    print("only a lib")




