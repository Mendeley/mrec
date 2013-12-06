"""
Make recommendations from a precomputed item similarity matrix.
"""

from recommender import ItemSimilarityRecommender

class PrecomputedItemSimilarityRecommender(ItemSimilarityRecommender):
    """
    Wrapper class to make recommendations using a precomputed item similarity matrix.

    Parameters
    ==========
    description : str
        Printable name for this recommender.
    similarity_matrix : array_like(num_items,num_items)
        The precomputed item similarity matrix.
    """


    def __init__(self,description,similarity_matrix):
        self.description = description
        self.set_similarity_matrix(similarity_matrix)

    def set_similarity_matrix(self,similarity_matrix):
        self.similarity_matrix = similarity_matrix

    def compute_similarities(self,j):
        return self.similarity_matrix[j,:]

    def fit(self,dataset,item_features=None):
        pass

    def __str__(self):
        return self.description
