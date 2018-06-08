def init_parameters():


    global vocab_size
    global encoding
    global hidden_dim
    global Word2Vec_embedding_dim
    global Word2Vec_batch_size
    global Word2Vec_optimizer_step
    global Word2Vec_window_size

    hidden_dim = 10
    vocab_size = 2000
    encoding = 'latin1'

    Word2Vec_embedding_dim = 10
    Word2Vec_batch_size = 256
    Word2Vec_optimizer_step = 0.01
    Word2Vec_window_size = 2

    return None
