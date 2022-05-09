import re
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
token_type = 'word'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import LambdaCallback

def lstm(txt, lenout):
    text = txt

    seq_length = 100
    start_story = '| ' * seq_length
    text = start_story + text
    text = text.lower()
    text = text.replace('\n\n\n\n\n', start_story)
    text = text.replace('\n', ' ')
    text = re.sub('  +', '. ', text).strip()
    text = text.replace('..', '.')
    text = re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)

    if token_type == 'word':
        tokenizer = Tokenizer(char_level = False, filters = '')
    else:
        tokenizer = Tokenizer(char_level = True, filters = '', lower = False)
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    token_list = tokenizer.texts_to_sequences([text])[0]


    def generate_sequences(token_list, step): 
        X = []
        y = []
        for i in range(0, len(token_list) - seq_length, step):
            X.append(token_list[i: i + seq_length])
            y.append(token_list[i + seq_length])
        y = to_categorical(y, num_classes = total_words)
        num_seq = len(X)
        print('Number of sequences:', num_seq, "\n")
        return X, y, num_seq

    step = 1
    seq_length = seq_length

    X, y, num_seq = generate_sequences(token_list, step)

    X = np.array(X)
    y = np.array(y)

    n_units = 256
    embedding_size = 100

    text_in = Input(shape = (None,))
    embedding = Embedding(total_words, embedding_size)
    x = embedding(text_in)
    x = LSTM(n_units)(x)
    # x = Dropout(0.2)(x)
    text_out = Dense(total_words, activation = 'softmax')(x)

    model = Model(text_in, text_out)

    opti = RMSprop(lr = 0.005)
    model.compile(loss='categorical_crossentropy', optimizer=opti)

    #model.summary()

    def sample_with_temp(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_text(seed_text, next_words, model, max_sequence_len, temp):
        output_text = seed_text
        
        seed_text = start_story + seed_text
        
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = token_list[-max_sequence_len:]
            token_list = np.reshape(token_list, (1, max_sequence_len))
            
            probs = model.predict(token_list, verbose=0)[0]
            y_class = sample_with_temp(probs, temperature = temp)
            
            if y_class == 0:
                output_word = ''
            else:
                output_word = tokenizer.index_word[y_class]
                
            if output_word == "|":
                break
                
            if token_type == 'word':
                output_text += output_word + ' '
                seed_text += output_word + ' '
            else:
                output_text += output_word + ' '
                seed_text += output_word + ' '             
        return output_text

    def on_epoch_end(epoch, logs):
        seed_text = ""
        gen_words = 300

        #print('Temp 0.2')
        generate_text(seed_text, gen_words, model, seq_length, temp = 0.2)
        #print('Temp 0.33')
        generate_text(seed_text, gen_words, model, seq_length, temp = 0.33)
        #print('Temp 0.5')
        generate_text(seed_text, gen_words, model, seq_length, temp = 0.5)
        #print('Temp 1.0')
        generate_text(seed_text, gen_words, model, seq_length, temp = 1)

    epochs = 10
    batch_size = 32
    num_batches = int(len(X) / batch_size)
    callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks = [callback], shuffle = True)

    def generate_human_led_text(model, max_sequence_len):
    
        output_text = ''
        seed_text = start_story
        
        from random import randint, random
        from tqdm.notebook import tqdm
        for _ in tqdm(range(int(lenout))):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = token_list[-max_sequence_len:]
            token_list = np.reshape(token_list, (1, max_sequence_len))
            
            probs = model.predict(token_list, verbose=0)[0]

            top_10_idx = np.flip(np.argsort(probs)[-10:])
            top_10_probs = [probs[x] for x in top_10_idx]
            top_10_words = tokenizer.sequences_to_texts([[x] for x in top_10_idx])

            r = random()
            if  1 > r > 1 - top_10_probs[0]:
                index = 0
            elif 1 - top_10_probs[0] > r > 1 - top_10_probs[0] - top_10_probs[1]:
                index = 1
            elif 1 - top_10_probs[0] - top_10_probs[1] > r > 1 - top_10_probs[0] - top_10_probs[1] - top_10_probs[2]:
                index = 2
            else:
                index = 3
                print(f"outlier!: {index}, {r}")

            chosen_word = top_10_words[index]
                
            
            seed_text += chosen_word + ' '
            output_text += chosen_word + ' '

        print (output_text)
        return output_text

    t = generate_human_led_text(model, seq_length)

    return t


if __name__ == "__main__":
    print(lstm("", 0))