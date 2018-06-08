from disintegrator import *
from Word2Vec import *
from seq2seq import *
from progress_bar import *
from parameters import *

init_parameters()


if __name__ == "__main__":

    import glob
    import os

    # INIT TRAINING MODEL
    text = ''
    for filename in glob.glob(os.path.join('./data/', '*.txt')):
        with open(filename, 'r', encoding=encoding) as file_obj:
            text = text + '. ' + file_obj.read()

    with open('data/stop_words', 'r', encoding=encoding) as file_obj:
        stopwords = file_obj.readlines()

    # CREATE A DICC AND PROCESSING DATA
    prepare = DataPreparation()
    text = prepare.make_disintegration(text)
    sent = prepare.get_sentences(text)
    dicc = prepare.get_dictionary(text, stopwords, vocab_size)
    data = prepare.get_word_list(sent, stopwords, window_size=Word2Vec_window_size)

    # WORD EMBEDDING ALGORITHM
    print('- TRAIN WORD EMBEDDING ALGORITHM \n')
    word_to_vec = Word2Vec(vocab_size, Word2Vec_embedding_dim, Word2Vec_optimizer_step)
    x_train, y_train = word_to_vec.training_data(data)
    _ = word_to_vec.train(x_train, y_train, Word2Vec_batch_size)

    conversations = []
    for i in range(len(sent) - 2):
        if len(sent[i+1]) != 0 and len(sent[i+2]) != 0:  # to avoid empty sentences
            conversations.append([sent[i+1], sent[i+2]])

    # TRAIN THE MODEL
    print('- TRAIN SEQ2SEQ ALGORITHM')
    printProgressBar(0, len(conversations), prefix='Progress:', suffix='Complete', length=50)

    with tf.Session() as session:

        saver = tf.train.Saver()
        init_variables = tf.global_variables_initializer()
        session.run(init_variables)
        losses = []

        for i, conv in enumerate(conversations):

            # Convert text to vector
            _input_sentence = word_to_vec.encoder(conv[0])
            _output_sentence = word_to_vec.encoder(conv[1])

            # Convert text to vector
            _input_sentence = np.transpose(np.array(_input_sentence))
            _output_sentence = np.transpose(np.array(_output_sentence))

            # Run the graph
            _, _pred_predicted_output, _loss = session.run([train_step, pred_predicted_output, loss],
                                                           feed_dict={input_sentence: _input_sentence,
                                                                      output_sentence: _output_sentence})

            losses.append(_loss)

            if i % (len(conversations) // 100) == 0:
                print('\n\n', 'Respuesta esperada: ', conv[1])
                print('Respuesta predicha: ', word_to_vec.decoder(_pred_predicted_output[:, 0:-1].tolist()), '\n')
                print('Loss: ', _loss)
                saver.save(session, "./model/seq2seq_iter", global_step=i)
                printProgressBar(i + 1, len(conversations), prefix='Progress:', suffix='Complete', length=50)

        # SAVE THE MODEL
        print('\n- SAVE THE MODEL')
        saver.save(session, "./model/seq2seq_model")

        # GRAPH VISUALIZATION

        '''
		This is needed to plot the graph in the tensorboar
		command: tensorboard --logdir="./model/seq2seq_graph_visualization"
		'''

        writer = tf.summary.FileWriter("./model/seq2seq_graph_visualization")
        writer.add_graph(session.graph)
