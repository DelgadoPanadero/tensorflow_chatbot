import sys
from disintegrator import *
from Word2Vec import *
from seq2seq import *

if __name__ == "__main__":

	# INPUT DATA
	_input_sentence = sys.argv[-1]
	prepare = DataPreparation()
	_input_sentence = prepare.make_disintegration

    # ENCODE THE SENTENCE
	word_to_vec = Word2Vec(vocab_size = vocab_size, embedding_dim = embedding_dim)
	_input_sentence = word_to_vec.encoder(' '.split(_input_sentence))
	_input_sentence = np.transpose(np.array(_input_sentence))

	# LOAD THE MODEL
	tf.reset_default_graph()
	saver = tf.train.import_meta_graph('./model/seq2seq_model.meta')

	# PROCESS THE ANSWER
	with tf.Session() as session:

		saver.restore(session,tf.train.latest_checkpoint('./model/'))
		graph = tf.get_default_graph()
		input_data = graph.get_tensor_by_name("input_data:0")
		predict_output = graph.get_tensor_by_name('predict_output:0')
#		predict_output = graph.get_tensor_by_name('whileloop_predict_sentence:0')
		_output_sentence = session.run(predict_output, feed_dict = {input_data : _input_sentence})


	# DECODE THE ANSWER
	_output_sentence = word_to_vec.decoder(_pred_predicted_output[:,0:-1].tolist())
	print(' '.join(_output_sentence))
