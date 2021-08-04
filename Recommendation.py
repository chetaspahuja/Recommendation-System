# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:15:45 2021

@author: ASUS
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import TruncatedSVD 

ratings_df = pd.read_csv('C:/GIT/Datasets/ratings_Beauty.csv')
ratings_df.info()
ratings_df.describe() 
ratings_df

ratings_df.shape

popular_products = pd.DataFrame(ratings_df.groupby('ProductId')['Rating'].count())
# popular_products = pd.DataFrame(ratings_df.groupby('ProductId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending = False)
print(popular_products)
print(most_popular)

most_popular.head(10)
most_popular.head(30).plot(kind = "bar") 
    


