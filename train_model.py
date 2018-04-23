from disintegrator import *
from word2vec import *
from seq2seq import *





if __name__ == "__main__":


	print('Init training model')

	with open('data/caperucita_roja', 'r') as file_obj: #encoding="ISO-8859-1"
    		text = file_obj.read()
    
	with open('data/stop_words', 'r') as file_obj:
   		 stopwords = file_obj.readlines()






	print('Created dicc and data')

	vocab_size = 200
	embedding_dim = 10


	prepare = data_preparation()

	text = prepare.make_disintegration(text)
	sent = prepare.get_sentences(text)
	dicc = prepare.get_dictionary(text, stopwords, vocab_size)
	data = prepare.get_word_list(sent, stopwords,window_size = 1)






	print('Created dicc and data')

	word_to_vec = word2vec(vocab_size, embedding_dim)
	x_train,y_train = word_to_vec.training_data(data)
	_ = word_to_vec.train(x_train,y_train)


	conversations = []

	for i in range(len(sent)-2):
    
    		if(len(sent[i+1]) != 0 and len(sent[i+2]) != 0): #to avoid empty sentences
        		conversations.append([sent[i+1],sent[i+2]]) 




	print('Train the model:')

	# Initialize all the variables
	session = tf.Session()
	init_variables = tf.global_variables_initializer()
	session.run(init_variables)
	losses = []


	for id, conv in enumerate(conversations):
   

    		# Convert text to vector
    		_input_sentence = word_to_vec.encoder(conv[0])
    		_output_sentence = word_to_vec.encoder(conv[1])


    		# Convert text to vector
    		_input_sentence = np.transpose(np.array(_input_sentence))
    		_output_sentence = np.transpose(np.array(_output_sentence))


   		 # Run the graph
    		_, _loss = session.run([train_step, loss], feed_dict={input_sentence : _input_sentence, output_sentence: _output_sentence})
    

    		losses.append(_loss)

    		if (id % (len(conversations)//100) == 0):

        		print('\t Processed: %d %%' %(id / (len(conversations)/100)))
