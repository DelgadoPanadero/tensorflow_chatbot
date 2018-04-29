import pickle
import numpy as np
from sklearn.decomposition import PCA





def euclidean_dist(vector1, vector2):

    '''
    Returns the euclidean-distance between two one-dimensional-arrays. 
    '''

    return np.sqrt(np.sum((vector1-vector2)**2))





def max_radius(words):

    '''
    Returns the euclidean-distance of the farest word of the list respect the origin.
    It is important to define the concept of proximity in our embedding space (such as a distante equal to the max_radius/10 for example).
    '''

    origin = words[0,:]*0
    distances = np.apply_along_axis(euclidean_dist, 1, words, origin)
    max_radius = np.max(distances)

    return max_radius





def meansquare_radius(words):

    '''
    Returns the euclidean-distance of the mean square distance of the list of words respect the origin.
    It is important to define the concept of proximity in our embedding space (such as a distante equal to the max_radius/10 for example).
    '''

    origin = words[0,:]*0
    distances = np.apply_along_axis(euclidean_dist, 1, words, origin)
    meansquare_radius = np.sqrt(np.mean(distances**2))

    return meansquare_radius







def variance_distribution(words):

    '''
    Returns the variance explained for everyone of the principal components.
    It is important to check that all the avalible embeding_space capacity is being used.
    '''

    pca = PCA()
    components = pca.fit_transform(words)
    variances = pca.explained_variance_ratio_

    return variances







def wildcardpoint_density(words, radius=meansquare_radius):

    '''
    Returns the variance explained for everyone of the principal components.
    It is important to check that all the available embeding_space capacity is being used.
    '''

    wildcard = words[0,:]*0
    distances = np.apply_along_axis(euclidean_dist, 1, words, wildcard)
    near_radius = radius(words)/np.sqrt(words.shape[-1])
    near_words_idx = np.where(distances < near_radius)

    return near_words_idx






def get_nearest_words(word_string, words, word_to_vec, n = 5, radius = max_radius):

    '''
    Given a string with a words returns a string with the n closest words
    '''


    word = word_to_vec.encoder([word_string])
    distances = np.apply_along_axis(euclidean_dist, 1, words,word)

    near_radius = radius(words)/np.sqrt(words.shape[-1])
    near_words = words[np.where(distances < near_radius),:]

    near_string_words = word_to_vec.decoder(near_words[0])
    near_string_words = near_string_words[:n]

    return near_string_words
