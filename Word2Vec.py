import pickle
import numpy as np
import tensorflow as tf
from parameters import *

init_parameters()


class Word2Vec(object):

    """
    Object for implementing word2vec algorithm in a dataset with the requiered structure.

    Requires:
        - The dataset
        - Dictionary of words and indexes
        - Parameters

    Saves in local:
        - Tensorflow graph
        - Tensors W1 and b1 as a np.array for the encoder and decoder.
    """

    def __init__(self, vocab_size, embedding_dim, optimizer_step):
        
        """Feed forward neuralnetwork's parameters. Architecture with two hidden layers.
        The vector representation of the word is the tensor 'encoder'.

        The input are word vectors in one-hot-encoding representation.
        The tarjet are the window size words arround the input word.
        """

        # DIMENSIONS
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.optimizer_step = optimizer_step

        # NEURALNET
        self.input_data = tf.placeholder(tf.float32, shape=(None, vocab_size), name='input_data')
        self.output_data = tf.placeholder(tf.float32, shape=(None, vocab_size), name='output_data')

        self.W1 = tf.Variable(tf.random_normal([vocab_size, embedding_dim]), name='W1')
        self.b1 = tf.Variable(tf.random_normal([embedding_dim]), name = 'b1')
        self.vector = tf.add(tf.matmul(self.input_data,self.W1), self.b1, name='encoder')

        self.W2 = tf.Variable(tf.random_normal([embedding_dim, vocab_size]), name='W2')
        self.b2 = tf.Variable(tf.random_normal([vocab_size]), name='b2')
        self.prediction = tf.nn.softmax(tf.add( tf.matmul(self.vector, self.W2), self.b2), name='prediction')

        # OPTIMIZATION
        self.loss = tf.reduce_mean(tf.squared_difference(self.output_data, self.prediction))
        self.train_step = tf.train.GradientDescentOptimizer(self.optimizer_step).minimize(self.loss)
        
    def to_one_hot(self, data_point_index):
        """ Given a number (index) returns the one-hot-representation vector.

        :param data_point_index: number
        :return: one_hot_vector
        """
        temp = np.zeros(self.vocab_size)
        temp[data_point_index] = 1
        return temp
    
    def training_data(self, _data):
        
        """First it transforms the word data to the index representation.Then it transforms the index representation to
        one-hot-encoding representation.

        It works with training data structure ([[word, [word,word]],...]) and with predictive ([[word],...])

        :param _data:
        :return:
        """
        
        with open('./model/dicc.pkl','rb') as file:
            _dicc = pickle.load(file)
            dicc_w2i = _dicc['w2i']
            
        input_train = []
        output_train = []

        for data_word in _data:

            input_index = dicc_w2i[data_word[0]] if  data_word[0] in dicc_w2i.keys() else 0
            #el imput siempre es solo una palabra
            input_train.append(self.to_one_hot(input_index))
            
            output_index = []
            for word in np.array(data_word[1]).reshape(-1):
                #el output es más enrevesado porque puede ser una palabra o una lista de palabras
                output_index.append(dicc_w2i[word] if word in dicc_w2i.keys() else 0)
                
            #output_index = [dicc_w2i[word] for word in np.array(data_word[1]).reshape(-1)]
            # el output es más enrevesado porque puede ser una palabra o una lista de palabras
            output_train.append(self.to_one_hot(output_index))

        input_train = np.asarray(input_train)
        output_train = np.asarray(output_train)  

        return input_train, output_train

    def train(self, _x_train, _y_train, batch_size=256):
        
        """Train the tensorflow graph.

        :param _x_train:
        :param _y_train:
        :param batch_size:
        :return:
        """
        
        n_data = len(_x_train)
        n_batch = n_data//batch_size

        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            
            for _ in range(n_batch):
                
                x = _x_train[(n_batch*batch_size):((n_batch+1)*batch_size)]
                y = _y_train[(n_batch*batch_size):((n_batch+1)*batch_size)]

                sess.run([self.train_step,self.vector], feed_dict={self.input_data: x, self.output_data: y})
                
                #if (batch_index+1)%(n_batch//100)==0:
                #    print('Progress: ', batch_index//n_batch*100)
            # saver = tf.train.Saver()
            # saver.save(sess, "./model/Word2Vec_model")
            
            _W1 = sess.run(self.W1)
            _b1 = sess.run(self.b1)
            
            np.save('model/Word2Vec_W1.npy', _W1)
            np.save('model/Word2Vec_b1.npy', _b1)
        
        return _W1, _b1
        
    def encoder(self, words):
        """Load the save graph and execute for the words in one-hot-representation.

        :param words: list of words to encode.
        :return: list of vector-lists.
        """

        '''
        with tf.Session() as sess:
            
            saver = tf.train.import_meta_graph('./model/Word2Vec_model.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./model/Word2Vec'))

            graph = tf.get_default_graph()
            
            input_data = graph.get_tensor_by_name("input_data:0")
            output_data = graph.get_tensor_by_name("output_data:0")
            vector = graph.get_tensor_by_name("vector:0")
                
            x_train = self.prediction_data(word)    
            vector = sess.run(vector, feed_dict={input_data: x_train})
        '''
        
        _W1 = np.load('model/Word2Vec_W1.npy')
        _b1 = np.load('model/Word2Vec_b1.npy')
        
        with open('./model/dicc.pkl', 'rb') as file:
            _dicc = pickle.load(file)
        
        dicc_w2i = _dicc['w2i']
        
        indexes = [dicc_w2i[word] if word in dicc_w2i else 0 for word in words]
        input_data = [self.to_one_hot(index) for index in indexes]
        input_data = np.reshape(input_data,(-1, self.vocab_size))
        vectors = np.dot(input_data,_W1)+_b1
        
        return vectors.tolist()     

    def decoder(self, vectors):
        
        """Returns the nearest word in the word-representation for the given vectors. It loads the graph, extract the
        tensors W1 and b1.

        :param vectors: list of vector-lists
        :return: list of words
        """
        
        _W1 = np.load('model/Word2Vec_W1.npy')
        _b1 = np.load('model/Word2Vec_b1.npy')
        
        with open('./model/dicc.pkl', 'rb') as file:
            _dicc = pickle.load(file)
        
        dicc_i2w = _dicc['i2w']
    
        def euclidean_dist(vector1, vector2):
            return np.sqrt(np.sum((vector1-vector2)**2))
        
        '''with tf.Session() as sess:

            saver = tf.train.import_meta_graph('./model/Word2Vec_model.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./model/model_Word2Vec'))
   
            graph = tf.get_default_graph()
            input_data = graph.get_tensor_by_name("input_data:0")
            output_data = graph.get_tensor_by_name("output_data:0")
            vocab_vector = graph.get_tensor_by_name("vector:0")        
        
        '''
        _vocab_vectors = _W1+_b1

        words = []
        for vector in vectors:
            
            distances = np.apply_along_axis(euclidean_dist, 1, _vocab_vectors, vector)
            nearest_index = np.argmin(distances)
            nearest_word = dicc_i2w[nearest_index] if nearest_index != 0 else ''
            words.append(nearest_word)
            
        return words

if __name__ == "__main__":

    from disintegrator import *
    from Word2Vec_topology import *
    import glob
    import sys
    import os
    import random

    text = ''

    # Si no pasamos argumentos por línea de comandos, leemos todos los archivos de ./data/
    if sys.argv[-1] == os.path.basename(__file__):

        for filename in glob.glob(os.path.join('./data/', '*.txt')):
            with open(filename, 'r', encoding=encoding) as file_obj:
                text = text + '. ' + file_obj.read()

    # Si pasamos nombres de archivos de ./data/ lee el nombre de los archivos que le pasemos
    else:
        for filename in sys.argv[1:]:
            filename = './data/' + filename
            with open(filename, 'r', encoding=encoding) as file_obj:
                text = text + '. ' + file_obj.read()

    # Stopwords
    with open('data/stop_words', 'r') as file_obj:
        stopwords = file_obj.readlines()

    print('Propiedades del texto: \n')
    print('\tTexto con %d caracteres' % (len(text)))
    print('\tTexto con %d palabras' % (len(text.split())))
    print('\n')

    prepare = DataPreparation()
    text = prepare.make_disintegration
    sent = prepare.get_sentences(text)
    dicc = prepare.get_dictionary(text, stopwords, vocab_size)
    training_data = prepare.get_word_list(sent, stopwords, window_size=Word2Vec_window_size)

    print('Propiedades del corpus: \n')
    print('\tDiccionario con %d palabras' %(len(dicc['w2i'])))

    word_to_vec = Word2Vec(vocab_size, embedding_dim, Word2Vec_window_size)
    x_train, y_train = word_to_vec.training_data(training_data)
    W1, b1 = word_to_vec.train(x_train, y_train)
    vocab_vectors = W1+b1

    print('Espacio embebido:\n')
    print('\tRadio máximo del espacio: %d' %max_radius(vocab_vectors))
    print('\tRadio medio del espacio: %d' %meansquare_radius(vocab_vectors))
    print('\tVarianza explicada dimensionalmente:')
    variances = variance_distribution(vocab_vectors)
    for i in range(embedding_dim):
        print('\t\t', variances[i])
    #print('\tPalabras entorno a la posición comodín: %d' %wildcardpoint_density(vocab_vectors))
    print('\tEjemplos:')
    word_idx = random.sample(range(len(data)),50)
    for idx in word_idx:
        print('\t\t', data[idx][0],': ', get_nearest_words(data[idx][0],vocab_vectors,word_to_vec))

    # vectors = model.encoder(['caperucita','lobo','abuela'])
    # palabras = model.decoder(vectors)
    # print(palabras)

