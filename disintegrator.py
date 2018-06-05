# coding=latin1
import re
import pickle
import collections
from unidecode import unidecode
from parameters import *

init_parameters()


class DataPreparation(object):

    @staticmethod
    def make_disintegration(_text):

        """The main object is to convert a text to a "plain text" with only lower letters and stops.

        :param  _text: real text
        :return:  plain text
        """

        _text = re.sub(r'\n+', '\n', _text)
        _text = re.sub(r'<.*?>', ' ', _text)
        _text = re.sub('^\\[a-zA-Z]*', ' ', _text)

        _text = re.sub(r'[,;\n—-“”:\"]', '.', _text)
        _text = re.sub(r'[?¿!¡]', '.', _text)
        _text = re.sub(r'[)(]', '.', _text)
        _text = re.sub(r' \.', '.', _text)

        _text = re.sub(r'\.+', '. ', _text)
        _text = re.sub(' +|\t', ' ', _text)
        
        return _text.lower()

    @staticmethod
    def get_sentences(_text):

        """Given a plain text return a list with all the sentences.

        :param _text: plain text with only lower letters and stops.
        :return: list of sentences from the text.
        """
        
        sentences = []

        for sentence in _text.split('.'):
            sentences.append(sentence.split())
            
        return sentences

    @staticmethod
    def get_dictionary(_text, stop_words, vocab_size):

        """This is made for getting an index-representation for the words in the text.
        It only creates an index for the "vocab_size" most popular words in the text.

        :param _text: plain text with only lower letters and stops.
        :param stop_words: stop_words' list.
        :param vocab_size: number of unique words for the dictionary.
        :return: dictionary: mapping word.
        """
        
        words = []
        
        for word in _text.split(' '):
            word = re.sub(r'\.', '', word)  # con esto quitamos el punto de la �ltima palabra en cada frase.

            if (word not in stop_words) and (re.match('^[a-zA-Z]*$', unidecode(word))) and (word != ''):
                words.append(word)

        count = collections.Counter(words).most_common(vocab_size-1)  # el -1 es porque para guardar dentro del
        # vocabulario un espacio para las palabras desconocidas.

        dicc_w2i = dict([(counter[0], index+1) for (index, counter) in enumerate(count)])  # el index+1 es para reservar
        #  el �ndice 0 para las palabras desconocidas.

        dicc_i2w = dict([(index+1, counter[0]) for (index, counter) in enumerate(count)])
        _dicc = {'w2i': dicc_w2i, 'i2w': dicc_i2w}

        with open("model/dicc.pkl", "wb") as file:
            pickle.dump(_dicc, file)

        return _dicc

    @staticmethod
    def get_word_list(sentences, stop_words, window_size=2):
        """Given a list of sentences, it makes a list with each word and the "window_size" words around.

        :param sentences: list of sentences. Each sentence is a list of words [['word1 word2 word3...'], '...', ...].
        :param stop_words: list of stopwords.
        :param window_size: number of words arround to take.
        :return: list of pairs for each word as [...,[word_n,[words arround word_n]],...].
        """

        word_list = []

        for sentence in sentences:   
            sentence = [word for word in sentence if ((word.lower() not in stop_words) and (re.match('^[a-zA-Z]*$',
                                                                                                     unidecode(word))))]

            for word_index, word in enumerate(sentence):
                word = word.lower()
                neighbourhood_words = sentence[max(word_index - window_size, 0):
                                               min(word_index + window_size, len(sentence)) + 1]
                neighbourhood_words = [neighbour.lower() for neighbour in neighbourhood_words
                                       if neighbour.lower() != word]

                while len(neighbourhood_words) < (2*window_size):
                    neighbourhood_words.append(word)

                    word_list.append([word, neighbourhood_words])
                    
        return word_list


if __name__ == "__main__":

    train_text = ''
    for filename in glob.glob(os.path.join('./data/', '*.txt')):

        with open(filename, 'r', encoding = encoding) as file_obj:
            train_text = train_text + '. ' + file_obj.read()
    
    with open('data/stop_words', 'r') as file_obj:
        stopwords = file_obj.readlines()

    prepare = DataPreparation()
    train_text = prepare.make_disintegration
    sent = prepare.get_sentences(train_text)
    dicc = prepare.get_dictionary(train_text, stopwords, vocab_size)
    data = prepare.get_word_list(sent, stopwords, window_size=Word2Vec_window_size)
    print('Created dictionary and data')

