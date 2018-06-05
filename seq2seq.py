import tensorflow as tf
from parameters import *

init_parameters()


class GRU(object):

    def __init__(self, input_dimensions, hidden_size, name='', dtype=tf.float64):

        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size
        self.name = name

        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Wr = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01),
            name='Wr' + self.name)
        self.Wz = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01),
            name='Wz' + self.name)
        self.Wh = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01),
            name='Wh' + self.name)

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Ur = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01),
            name='Ur' + self.name)
        self.Uz = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01),
            name='Uz' + self.name)
        self.Uh = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01),
            name='Uh' + self.name)

        # Biases for hidden vectors of shape (hidden_size,)
        self.br = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01),
                              name='br' + self.name)
        self.bz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01),
                              name='bz' + self.name)
        self.bh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01),
                              name='bh' + self.name)

    def forward_pass(self, h_tm1, x_t):  # Function though to be used by tf.scan

        """Perform a forward pass.
        
        Arguments
        ---------
        h_tm1: np.matrix. The hidden state at the previous timestep (h_{t-1}).
        x_t: np.matrix. The input vector.
        """

        # Convert vector-tensor form into  matrix-tensor form
        x_t = tf.reshape(x_t, shape=[1, -1])
        h_tm1 = tf.reshape(h_tm1, shape=[1, -1])

        # Definitions of z_t and r_t
        z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
        r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)

        # Definition of h~_t
        h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)

        # Compute the next hidden state
        h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

        return tf.squeeze(h_t)

    def process_sequence(self, sequence, h_0=None):

        # Put the time-dimension upfront for the scan operator
        self.x_t = tf.transpose(sequence, [1, 0], name='x_t')  # [n_words, embedding_dim]

        if h_0 is None:
            # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
            self.h_0 = tf.zeros(dtype=tf.float64, shape=(self.hidden_size,), name='h_0')
        else:
            self.h_0 = h_0

        # Perform the scan operator (hacky as fac diud)
        self.h_t_transposed = tf.scan(self.forward_pass, self.x_t, self.h_0, name='h_t_transposed')

        # Transpose the result back
        self.h_t = tf.transpose(self.h_t_transposed, [1, 0], name='h_t')

        return self.h_t

    def predict_sequence(self, sequence, h_0):

        '''
        Output sequence. This function iterates self.forward_pass until it gets the EOL.
        '''

        # Inital values. The are required to be reshaped to rank2-tensor be concated afterwards
        init_predict_sentence = tf.zeros([10, 1], dtype=tf.float64, name='whileloop_init_sentence')
        init_prediction = tf.reshape(h_0, shape=[-1, 1], name='whileloop_init_prediction')

        def loop_cond(prediction, predict_sentence):
            threshold = tf.constant(0.01, dtype=tf.float64, name='whileloop_threshold')
            boolean = tf.greater((tf.reduce_sum(tf.pow(prediction, 2)) ** 0.5), threshold, name='whileloop_boolean')
            return boolean

        def loop_body(prev_prediction, prev_predict_sentence):

            '''
            This function is a little bit hacky. Tensorflow's loops don't support neither fetching global scope variables
            that are transformed but not returned from the loop nor modify the rank of the returned tensor in every
            iteration of the loop. 
            
            This seems to be overcome defining the predict_sentence in two stages, one for the previous iter state an
            another one for the next state.
            '''

            # In the predict_model the previous state and the input state for the forward_pass are the same
            next_prediction = self.forward_pass(prev_prediction, prev_prediction)
            next_prediction = tf.reshape(next_prediction, shape=[-1, 1], name='whileloop_next_prediction')

            # Concat the predicted word to the sentence (instead of list.append() cause tf.while_loop() doesn't support no-tensor arguments)
            next_predict_sentence = tf.concat(axis=1, values=[prev_prediction, prev_predict_sentence],
                                              name='whileloop_next_prediction_sentence')

            return [next_prediction, next_predict_sentence]

        # While loop that return the predict sentence
        _, predict_sentence = tf.while_loop(cond=loop_cond,
                                            body=loop_body,
                                            loop_vars=[init_prediction, init_predict_sentence],
                                            shape_invariants=[tf.TensorShape([10, 1]), tf.TensorShape([10, None])],
                                            maximum_iterations=10,
                                            name='whileloop_predict_sentence')
        return predict_sentence


# ### Initialize the model

# The input has 2 dimensions: dimension 0 is reserved for the first term and dimension 1 is reserved for the second term

# Create a placeholder
input_sentence = tf.placeholder(dtype=tf.float64, shape=[embedding_dim, None], name='input_data')  # emb_dim x n_words
output_sentence = tf.placeholder(dtype=tf.float64, shape=[embedding_dim, None], name='output_data')

# Create End Of Sentence vector
EOS = tf.zeros(dtype=tf.float64, shape=[embedding_dim, 1], name='EOS')
input_sentence_ended = tf.concat([input_sentence, EOS], axis=1, name='input_data_ended')
output_sentence_ended = tf.concat([output_sentence, EOS], axis=1, name='output_data_ended')

# Create the GRU layer
gru_layer_encoder = GRU(embedding_dim, hidden_dim, name='_encoder')
gru_layer_decoder = GRU(embedding_dim, hidden_dim, name='_decoder')

# Training_process - ONE NN ENCODER - DECODER
input_encoded = gru_layer_encoder.process_sequence(input_sentence_ended, h_0=None)  # Process the first sentence
thought_vector = input_encoded[:, -1]  # Extract the last state vector (thought) from the input response
train_decoded = gru_layer_decoder.process_sequence(output_sentence_ended, h_0=thought_vector)  # Train_answer
pred_decoded = gru_layer_decoder.predict_sequence(output_sentence_ended, h_0=thought_vector)

# Output_data
train_predicted_output = tf.convert_to_tensor(train_decoded, dtype=tf.float64, name='train_output')
pred_predicted_output = tf.convert_to_tensor(pred_decoded, dtype=tf.float64, name='pred_output')

# Loss
loss = tf.reduce_sum(0.5 * tf.pow(train_predicted_output - output_sentence_ended, 2))  # / float(batch_size)
# loss = [sum((real_word-prediction)**2)/embedding_dim for (real_word, prediction) in zip(real_words, predictions)]

# Optimizer
train_step = tf.train.AdamOptimizer().minimize(loss)

if __name__ == "__main__":

    from disintegrator import *
    from Word2Vec import *

    parameters.init()

    # Prepare data for training the seq2seq
    prepare = DataPreparation()
    text = prepare.make_disintegration
    sent = prepare.get_sentences(text)
    dicc = prepare.get_dictionary(text, stopwords, vocab_size)
    data = prepare.get_word_list(sent, stopwords, window_size=window_size)

    print('Propiedades del corpus: \n')
    print('\tDiccionario con %d palabras' %(len(dicc['w2i'])))

    word_to_vec = Word2Vec(vocab_size, embedding_dim)
    x_train, y_train = word_to_vec.training_data(data)
    W1, b1 = word_to_vec.train(x_train, y_train)
    vocab_vectors = W1+b1
    conversations = []

    for i in range(len(sent) - 2):
        if len(sent[i + 1]) != 0 and len(sent[i + 2]) != 0:  # to avoid empty sentences
            conversations.append([sent[i + 1], sent[i + 2]])

    # TRAIN THE MODEL

    # Initialize all the variables
    session = tf.Session()
    init_variables = tf.global_variables_initializer()
    session.run(init_variables)
    losses = []

    for conv in conversations:
        # Convert text to vector
        _input_sentence = word_to_vec.encoder(conv[0])
        _output_sentence = word_to_vec.encoder(conv[1])

        # Convert list-structure to array-structure
        _input_sentence = np.transpose(np.array(_input_sentence))
        _output_sentence = np.transpose(np.array(_output_sentence))

        # Run the graph
        _, _loss = session.run([train_step, loss], feed_dict={input_sentence: _input_sentence, output_sentence: _output_sentence})

        losses.append(_loss)

    # Save the model
    saver = tf.train.Saver()
    saver.save(sess, "./model/seq2seq_model")

    # Prediction
    _input_sentence = 'hola que tal?'
    print('yo: \t', _input_sentence)

    # Convert text to vector
    _input_sentence = word_to_vec.encoder(' '.split(_input_sentence))

    # Convert list-structure to array-structure
    _input_sentence = np.transpose(np.array(_input_sentence))

    # Run the graph
    pred = session.run(pred_predicted_output, feed_dict={input_sentence: _input_sentence})

    # Decode
    pred = np.transpose(np.array(pred))
    _output_sentence = word_to_vec.decoder(pred)

    # Sentence
    print('bot: \t', ' '.join(_output_sentence))
