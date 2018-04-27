import sys
import os
import glob
from disintegrator import *
from word2vec import *
from seq2seq import *
from progress_bar import *




if __name__ == "__main__":



	# INIT TRAINING MODEL
	print('- INIT TRAINING MODEL \n')

	text = ''

	for filename in glob.glob(os.path.join('./data/', '*.txt')):

		with open(filename, 'r') as file_obj:
    			text = text + '. ' + file_obj.read()
    
	with open('data/stop_words', 'r') as file_obj:
   		 stopwords = file_obj.readlines()





	# CREATE A DICC AND PROCESSING DATA
	print('- CREATE DICC AND PROCESSING DATA \n')

	vocab_size = 2000
	embedding_dim = 10


	prepare = data_preparation()

	text = prepare.make_disintegration(text)
	sent = prepare.get_sentences(text)
	dicc = prepare.get_dictionary(text, stopwords, vocab_size)
	data = prepare.get_word_list(sent, stopwords,window_size = 1)





	# WORD EMBBEDING ALGORITHM
	print('- WORD EMBBEDING ALGORITHM \n')

	word_to_vec = word2vec(vocab_size, embedding_dim)
	x_train,y_train = word_to_vec.training_data(data)
	_ = word_to_vec.train(x_train,y_train)


	conversations = []

	for i in range(len(sent)-2):
    
    		if(len(sent[i+1]) != 0 and len(sent[i+2]) != 0): #to avoid empty sentences
        		conversations.append([sent[i+1],sent[i+2]]) 





	#TRAIN THE MODEL
	print('- TRAIN THE MODEL')
	printProgressBar(0, len(conversations), prefix = 'Progress:', suffix = 'Complete', length = 50)


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
	    		_,_pred_predicted_output, _loss = session.run([train_step,pred_predicted_output, loss], feed_dict={input_sentence : _input_sentence, output_sentence: _output_sentence})
    

    			losses.append(_loss)

    			if (i % (len(conversations)//100) == 0):
        			print('\n\n',conv[1])
        			print(word_to_vec.decoder(_pred_predicted_output.shape),'\n')
        			saver.save(session, "./model/seq2seq_iter", global_step = i)
        			printProgressBar(i + 1, len(conversations), prefix = 'Progress:', suffix = 'Complete', length = 50)



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
