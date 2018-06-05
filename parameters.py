def init_parameters():

    global embedding_dim
    global vocab_size
    global encoding
    global hidden_dim
    global Word2Vec_batch_size
    global window_size

    embedding_dim = 10
    hidden_dim = 10
    vocab_size = 2000
    encoding = 'latin1'
    window_size = 2
    Word2Vec_batch_size = 256

    return None
