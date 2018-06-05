import re
import pickle
import collections
from unidecode import unidecode
from parameters import *

init_parameters()


class DataPreparation(object):

    @staticmethod
    def make_disintegration(text):

        """The main object is to convert a text to a "plain text" with only lower letters and stops.

        input :  real text
        output : plain text
        """

        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub('^\\[a-zA-Z]*', ' ', text)

        text = re.sub(r'[,;\n—-“”:\"]', '.', text)
        text = re.sub(r'[?¿!¡]', '.', text)
        text = re.sub(r'[)(]', '.', text)
        text = re.sub(r' \.', '.', text)

        text = re.sub(r'\.+', '. ', text)
        text = re.sub(' +|\t', ' ', text)
        
        return text.lower()

    @staticmethod
    def get_sentences(text):

        """Given a plain text return a list with all the sentences.

        text: plain text with only lower letters and stops.
        sentences: list of text chunks split by stops.
        """
        
        sentences = []

        for sentence in text.split('.'):
            sentences.append(sentence.split())
            
        return sentences

    @staticmethod
    def get_dictionary(text, stop_words, vocab_size):

        """This is made for getting an index-representation for the words in the text.
        It only creates an index for the "vocab_size" most popular words in the text.

        text: plain text with only lower letters and stops.
        dicc_w2i: mapping word
        """
        
        words = []
        
        for word in text.split(' '):
            word = re.sub(r'\.', '', word)  # con esto quitamos el punto de la última palabra en cada frase

            if (word not in stop_words) and (re.match('^[a-zA-Z]*$', unidecode(word))) and (word != ''):
                words.append(word)

        count = collections.Counter(words).most_common(vocab_size-1)  # el -1 es porque para guardar dentro del
        # vocabulario un espacio para las palabras desconocidas

        dicc_w2i = dict([(counter[0], index+1) for (index, counter) in enumerate(count)])  # el index+1 es para reservar
        #  el índice 0 para las palabras desconocidas

        dicc_i2w = dict([(index+1, counter[0]) for (index, counter) in enumerate(count)])
        dictionary = {'w2i': dicc_w2i, 'i2w': dicc_i2w}

        with open("model/dicc.pkl", "wb") as file:
            pickle.dump(dicc, file)
            
            
        return dictionary
    
    
    
    ''' 
    def get_word_word(self, sentences, stop_words, window_size = 2):
    
        data = []
    
        for sentence in sentences:
            sentence = [word for word in sentence if ((word.lower() not in stop_words) and (re.match('^[a-zA-Z]*$',unidecode(word))))]
    
     
            for word_index, word in enumerate(sentence):      
                neighbourhood_words = sentence[max(word_index - window_size, 0) : min(word_index + window_size, len(sentence)) + 1]
            
        
                for neighbour_word in neighbourhood_words:       
                    neighbour_word = neighbour_word.lower()
                    word = word.lower()      
            
            
                    if neighbour_word != word:
                        data.append([word, neighbour_word])
                                      
        return data
    '''

    @staticmethod
    def get_word_list(sentences, stop_words, window_size=2):

        """
        Given a list of sentences, it makes a list with each word and the "window_size" words around.

        sentence = ['word1 word2 word3...', '...', ...]
        word_list =  = [word2, [word1,word2]]
        """

        word_list = []

        for sentence in sentences:   
            sentence = [word for word in sentence if ((word.lower() not in stop_words) and (re.match('^[a-zA-Z]*$',
                                                                                                     unidecode(word))))]

            for word_index, word in enumerate(sentence):
                word = word.lower()
                neighbourhood_words = sentence[max(word_index - window_size, 0):
                                               min(word_index + window_size, len(sentence)) + 1]
                neighbourhood_words = [neighbour.lower() for neighbour in neighbourhood_words if neighbour.lower() != word]

                while len(neighbourhood_words) < (2*window_size):
                    neighbourhood_words.append(word)

                    word_list.append([word, neighbourhood_words])
                    
        return word_list


if __name__ == "__main__":

    import parameters

    train_text = ''
    for filename in glob.glob(os.path.join('./data/', '*.txt')):

        with open(filename, 'r') as file_obj:
            train_text = train_text + '. ' + file_obj.read()
    
    with open('data/stop_words', 'r') as file_obj:
        stopwords = file_obj.readlines()

    prepare = DataPreparation()
    train_text = prepare.make_disintegration
    sent = prepare.get_sentences(train_text)
    dicc = prepare.get_dictionary(train_text, stopwords, vocab_size)
    data = prepare.get_word_list(sent, stopwords, window_size=window_size)
    print('Created dicc and data')

