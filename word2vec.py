import pickle
import numpy as np
import tensorflow as tf



class word2vec(object):
    

    '''
    Object for implementing word2vec algorithm in a dataset with the requiered structure.
    
    Requires:
    
        - The dataset
        - Dictionary of words and indexes
        - Parameters
        
    Saves in local:
    
        - Tensorflow graph
        - Tensors W1 and b1 as a np.array for the encoder and decoder.
    '''
    

    def __init__(self,vocab_size,embedding_dim):
        
        '''
        Feed forward neuralnet architecture with two hidden layers.
        
        input: word vector in one-hot-encoding representation
        label: window size words
        
        The vector representation of the word is the tensor "encoder"
        '''
        
        # DIMENSIONS
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.optimizer_step = 0.01
        
    
        # NEURALNET
        self.input_data = tf.placeholder(tf.float32, shape=(None, vocab_size), name = 'input_data')
        self.output_data = tf.placeholder(tf.float32, shape=(None, vocab_size), name = 'output_data')

        
        self.W1 = tf.Variable(tf.random_normal([vocab_size, embedding_dim]), name = 'W1')
        self.b1 = tf.Variable(tf.random_normal([embedding_dim]), name = 'b1')
        self.vector = tf.add(tf.matmul(self.input_data,self.W1), self.b1, name = 'encoder')
        
        
        self.W2 = tf.Variable(tf.random_normal([embedding_dim, vocab_size]), name = 'W2')
        self.b2 = tf.Variable(tf.random_normal([vocab_size]), name = 'b2')
        self.prediction = tf.nn.softmax(tf.add( tf.matmul(self.vector, self.W2), self.b2), name = 'prediction')

        
        
        # OPTIMIZATION
        self.cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(self.output_data * tf.log(self.prediction), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(self.optimizer_step).minimize(self.cross_entropy_loss)   
    
        
        
        
        
    def to_one_hot(self, data_point_index):
        
        temp = np.zeros(self.vocab_size)
        temp[data_point_index] = 1
        return temp
    
    
    
    
    
    def training_data(self, data):
        
        '''
        First it transforms the word data to the index representation.
        Then it transforms the index representation to one-hot-encoding representation.
        
        It works with training data structure ([[word, [word,word]],...]) and with predictive ([[word],...])
        
        '''
        
        with open('./models/dicc.pkl','rb') as file:
            dicc = pickle.load(file)
            dicc_w2i = dicc['w2i']

            
            
        input_train = []
        output_train = []
        
        
        #if(len(data[0])==2): #if data is predictive data
        for data_word in data:
            
            #input_indexes = word2int[data_word[0]]            
            input_index = dicc_w2i[data_word[0]] if  data_word[0] in dicc_w2i.keys() else 0 #el imput siempre es solo una palabra
            input_train.append(self.to_one_hot(input_index))
            
            output_index = []
            for word in np.array(data_word[1]).reshape(-1):#el output es más enrevesado porque puede ser una palabra o una lista de palabras
                output_index.append(dicc_w2i[word] if word in dicc_w2i.keys() else 0)
                
            #output_index = [dicc_w2i[word] for word in np.array(data_word[1]).reshape(-1)] #el output es más enrevesado porque puede ser una palabra o una lista de palabras
            output_train.append(self.to_one_hot(output_index))

        
        '''elif(len(data[0])==1): #if data is training data
            
            for data_word in data:
            
                #input_indexes = word2int[data_word[0]]
                input_indexes = word2int[np.reshape(data_word,(1,-1))[0]] #el imput siempre es solo una palabra
                input_train.append(self.to_one_hot(input_indexes, self.vocab_size))  
              
        
        else: 
            continue
        '''
        
        input_train = np.asarray(input_train)
        output_train = np.asarray(output_train)  
        
        
        return (input_train,output_train)
    
    
    
    

    def train(self, x_train, y_train, batch_size = 256):
        
        '''
        Train the tensorflow graph.
        '''
        
        n_data = len(x_train)
        n_batch = n_data//batch_size
        
        
        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            
            for batch_index in range(n_batch):
                
                
                
                x = x_train[(n_batch*batch_size):((n_batch+1)*batch_size)]
                y = y_train[(n_batch*batch_size):((n_batch+1)*batch_size)]

                    
                sess.run([self.train_step,self.vector], feed_dict={self.input_data: x, self.output_data: y})
                
                #if (batch_index+1)%(n_batch//100)==0:
                #    print('Progress: ', batch_index//n_batch*100)
        
        
        
        
            saver = tf.train.Saver()
            saver.save(sess, "./models/word2vec_model")
            
            W1 = sess.run(self.W1)
            b1 = sess.run(self.b1)
            
            np.save('models/word2vec_W1.npy', W1)
            np.save('models/word2vec_b1.npy', b1)
        
        
        return (W1, b1)
    
        
        
    def encoder(self, words):
        
        '''
        Load the save graph and execute for the words in one-hot-representation
        '''
        
        '''
        with tf.Session() as sess:
            
            saver = tf.train.import_meta_graph('./models/word2vec_model.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./models/word2vec'))

            graph = tf.get_default_graph()
            
            input_data = graph.get_tensor_by_name("input_data:0")
            output_data = graph.get_tensor_by_name("output_data:0")
            vector = graph.get_tensor_by_name("vector:0")
                
            x_train = self.prediction_data(word)    
            vector = sess.run(vector, feed_dict={input_data: x_train})
        '''
        
        W1 = np.load('models/word2vec_W1.npy')
        b1 = np.load('models/word2vec_b1.npy')
        
        with open('./models/dicc.pkl','rb') as file:
            dicc = pickle.load(file)
        
        dicc_w2i = dicc['w2i']
        
        indexes = [dicc_w2i[word] if word in dicc_w2i else 0 for word in words]
        input_data = [self.to_one_hot(index) for index in indexes]
        input_data = np.reshape(input_data,(-1,self.vocab_size))
        vectors = np.dot(input_data,W1)+b1
        
        return vectors.tolist()     
           
    
    
    def decoder(self, vectors):
        
        '''
        Returns the nearest word in the word-representation for the given vectors.
        
        It loads the graph, extract the tensors W1 y b1 and 
        '''
        
        W1 = np.load('models/word2vec_W1.npy')
        b1 = np.load('models/word2vec_b1.npy')
        
        with open('./models/dicc.pkl','rb') as file:
            dicc = pickle.load(file)
        
        dicc_i2w = dicc['i2w']
    
        def euclidean_dist(vector1, vector2): return np.sqrt(np.sum((vector1-vector2)**2))   

        
        '''with tf.Session() as sess:

            saver = tf.train.import_meta_graph('./models/word2vec_model.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./models_word2vec'))
   
            graph = tf.get_default_graph()
            input_data = graph.get_tensor_by_name("input_data:0")
            output_data = graph.get_tensor_by_name("output_data:0")
            vocab_vector = graph.get_tensor_by_name("vector:0")        
        
        '''
        vocab_vectors = W1+b1
        

        words = []
        
        for vector in vectors:
            
            distances = np.apply_along_axis(euclidean_dist, 1, vocab_vectors, vector)
            nearest_index = np.argmin(distances)
            nearest_word = dicc_i2w[nearest_index] if nearest_index!=0 else ''
            
            words.append(nearest_word)
            
        return words





if __name__ == "__main__":


	model = word2vec(vocab_size, embedding_dim)
	x_train,y_train = model.training_data(data)
	_ = model.train(x_train,y_train)


	vectors = model.encoder(['caperucita','lobo','abuela'])
	palabras = model.decoder(vectors)
	print(palabras)

