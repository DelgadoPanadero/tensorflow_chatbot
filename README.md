# CHATBOT

This project consist on a AI chatbot training throught out plain text which is located in `./data`. The project is developed in different modules. The main script are `./train_model.py` which execute all the reading. This scripts preforms the execution of all the needed modules.

## Modules

### Disintegrator.py

It performs the text mining and text preparation moreover it creates and saves a diccionary with the most common words in te file `/models/dicc.pkl`

### Word2vec.py

Word2vec algorithm from scratch. It consist on a tensorflow's neuralnet which convert the words from a plain into vectors in a embbeding space with a defined dimension. It requires a diccionary such as `/models/dicc.pkl` with the most common words.

The main reason to build it is because i did not find any documentation about word2vec decoding in the existing libraries.

### Seq2seq

Seq2seq algorithm from scratch with GRU nets in tensorflow. It receives a plain text and creates a sequence of question/answer pairs. The script iters over this sequence and execute the word2vecencoding word by word and pass this to the neuralnet.

I have developed this module instead of using an existing one because this model in sensible to the way in which it is implemented and usually they use ad hoc performance.

## TODO

* Create an attention mechanism
* Create a webapp to deploy the model in a chat.
* Process either in GPU or optimize GPU parallelize
