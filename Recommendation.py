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

# Grouping data according to number of feedback/rating in the dataset
popular_products = pd.DataFrame(ratings_df.groupby('ProductId')['Rating'].count())

# Sorting the product list with most rated product on top
most_popular = popular_products.sort_values('Rating', ascending=False)

# print(popular_products)
# print(most_popular)

# taking top 10 most rated products
most_popular.head(10)

# plotting for top 30 most rated products

most_popular.head(30).plot(kind="bar")


# THE ABOVE HELPS TO RECOMMEND THE PRODUCT TO A NEW USER WITH NO VIEW HISTORY
# --------------------------------------------------------------------------

# Model-based collaborative filtering system
# Recommend items to users based on purchase history and similarity of ratings provided by other users who bought items to that of a particular customer.

ratings_1 = ratings_df.head(10000)
ratings_df_utility_matrix = ratings_1.pivot_table(
    values='Rating', index='UserId', columns='ProductId', fill_value=0)
# print(ratings_df_utility_matrix.head())

ratings_df_utility_matrix.shape

trans = ratings_df_utility_matrix.T
trans.head()

trans.shape

trans_1 = trans

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(trans)
decomposed_matrix.shape

correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape
# print(correlation_matrix)

# ISOLATING A PRODUCT ID

i = trans.index[9]

product_name = list(trans.index)
product_ID = product_name.index(i)
product_ID

correlation_product_ID = correlation_matrix[product_ID]
# print(correlation_product_ID > 0.9)

# correlation_product_ID.shape

Recommend = list(trans.index[correlation_product_ID > 0.90])

# REMOVES THE ITEM ALREADY BOUGHT BY THE USER AS CORRELATION  = 1
Recommend.remove(i)
print("LIST")
print(Recommend[0:9]) 

# -------------------------------------------------------------------------------

# Importing libraries 
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer 
from sklearn.neighbors import NearestNeighbors 
from sklearn.cluster import KMeans 
from sklearn.metrics import adjusted_rand_score 

# RECOMMENDING PRODUCTS BASED ON DESCRIPTION 
product_description = pd.read_csv('C:\GIT\Datasets\product_descriptions.csv')
# product_description.info() 
product_description.shape

#Checking for empty values , dropna -> removes missing values 
product_description = product_description.dropna() 
product_description.shape 

product_description.head() #By defualt head(5) -> top 5 most values 

#  Converting the text into numerical for data analysis 
vectorizer = TfidfVectorizer(stop_words = 'english') 
X1 = vectorizer.fit_transform(product_description["product_description"]) 
# print(X1) 


X=X1

kmeans = KMeans(n_clusters = 10, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)
plt.plot(y_kmeans, ".")
plt.show()

# Reduce the dataset , takes long time to process 
