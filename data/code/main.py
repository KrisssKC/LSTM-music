from tqdm import *

def get_lenout(data, ch):
    d = data[ch].meta.split('\n')[0]
    return d[1], d[2]

def train_note_channel(data, ch):
    notelen = len(data[ch].no_note)
    data[ch].no_note =  lstm(data[ch].no_note, int(notelen))

def fast_train(filename):
    data, n, speed, tempo, nom, denom, key= preprocess(filename)
    for i in (range(1)):
        train_note_channel(data, i)

    postprocessing(data, n, speed, tempo, nom, denom, key)
    

fast_train('HotelCalifornia.mid')



