import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import *


'''
Loading all the numpy files containing the utterance information and text information
'''


def load_data():
    speech_train = np.load('train_new.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('dev_new.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('test_new.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('dev_transcripts.npy', allow_pickle=True, encoding='bytes')

    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid


'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter2index
'''


def transform_letter_to_index(transcript, letter2index):
    """
    :param transcript :(N, ) Transcripts are the text input
    :param letter2index: letter2index dict
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    """
    full_res = []

    for cur_sentence in transcript:
        cur_res = [letter2index["<sos>"], ]
        
        for cur_word in cur_sentence:
            cur_res += [letter2index[c] for c in cur_word.decode("utf-8")]
            cur_res.append(letter2index[" "])

        # pop the last space
        cur_res.pop()
        cur_res.append(letter2index["<eos>"])

        full_res.append(np.array(cur_res))
            
    return np.array(full_res)


def transform_index_to_letter(index_arr, letter_list):
    """
    :param index_arr :(N, ) index
    :param letter_list: index2index dict
    :return transcript:
    """
    transcript = "".join([letter_list[i] for i in index_arr[1:-1]])

    return transcript


'''
Optional, create dictionaries for letter2index and index2letter transformations
'''


def create_dictionaries(letter_list):
    letter2index = {y: x for x, y in enumerate(letter_list)}
    index2letter = {x: y for x, y in enumerate(letter_list)}
    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    """
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours.
    """
    def __init__(self, speech, text=None, is_train=True):
        self.speech = speech
        self.speech_len = [len(x) for x in speech]

        self.is_train = is_train
        if text is not None:
            self.text = text
            self.text_len = [len(x) for x in text]

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if self.is_train:
            return torch.tensor(self.speech[index].astype(np.float32)), self.speech_len[index], \
                   torch.tensor(self.text[index]), self.text_len[index] - 1,
        else:
            return torch.tensor(self.speech[index].astype(np.float32)), self.speech_len[index]


def collate_train(batch_data):
    # Return the padded speech and text data, and the length of utterance and transcript ###
    cur_speech, cur_speech_len, cur_text, cur_text_len = zip(*batch_data)
    cur_speech = pad_sequence(cur_speech)
    cur_text = pad_sequence(cur_text, batch_first=True)

    return cur_speech, torch.tensor(cur_speech_len, dtype=torch.int64), \
           cur_text, torch.tensor(cur_text_len, dtype=torch.int64)


def collate_test(batch_data):
    # Return padded speech and length of utterance ###
    cur_speech, cur_speech_len = zip(*batch_data)
    cur_speech = pad_sequence(cur_speech)

    return cur_speech, torch.tensor(cur_speech_len, dtype=torch.int64)
