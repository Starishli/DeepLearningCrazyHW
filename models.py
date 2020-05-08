import torch
import torch.nn as nn
import torch.nn.utils as utils
import numpy as np

from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SOS_INDEX = 33
EOS_INDEX = 34


class Attention(nn.Module):
    """
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    """
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        """
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, key_size) Key Projection from Encoder per time step
        :param value: (N, value_size) Value Projection from Encoder per time step
        :param lens: (N)
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted
        """
        key = torch.transpose(key, 0, 1)
        value = torch.transpose(value, 0, 1)
        query = query.unsqueeze(2)

        mask = (torch.arange(value.shape[1]).reshape((-1, 1)) >= lens).transpose(0, 1).to(DEVICE)

        energy = torch.bmm(key, query).squeeze(2)
        energy.masked_fill_(mask, -1e9)
        energy = energy.unsqueeze(2)

        attention = torch.softmax(energy, dim=1)

        attention = torch.transpose(attention, 1, 2)

        context = torch.bmm(attention, value).squeeze(1)

        return context, mask


class pBLSTM(nn.Module):
    """
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    """
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=hidden_dim * 4, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.lstm_3 = nn.LSTM(input_size=hidden_dim * 4, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

        self.lstm_layers = nn.ModuleList([self.lstm_1, self.lstm_2, self.lstm_3])

    def forward(self, x):
        """
        :param x :(T, N, H) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM
        """

        x, lens = utils.rnn.pad_packed_sequence(x, batch_first=False)

        x = torch.transpose(x, 0, 1)
        x = x[:x.shape[0], :x.shape[1] // 2 * 2, :]
        x = x.reshape((x.shape[0], x.shape[1] // 2, x.shape[2] * 2))
        x = torch.transpose(x, 0, 1)
        lens = lens // 2

        cur_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        x, _ = self.lstm_1(cur_inp)

        x, lens = utils.rnn.pad_packed_sequence(x, batch_first=False)

        x = torch.transpose(x, 0, 1)
        x = x[:x.shape[0], :x.shape[1] // 2 * 2, :]
        x = x.reshape((x.shape[0], x.shape[1] // 2, x.shape[2] * 2))
        x = torch.transpose(x, 0, 1)
        lens = lens // 2

        cur_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        x, _ = self.lstm_2(cur_inp)

        x, lens = utils.rnn.pad_packed_sequence(x, batch_first=False)

        x = torch.transpose(x, 0, 1)
        x = x[:x.shape[0], :x.shape[1] // 2 * 2, :]
        x = x.reshape((x.shape[0], x.shape[1] // 2, x.shape[2] * 2))
        x = torch.transpose(x, 0, 1)
        lens = lens // 2

        cur_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        x, _ = self.lstm_3(cur_inp)

        return x


class Encoder(nn.Module):
    """
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    """
    def __init__(self, input_dim, hidden_dim, value_size, key_size):
        super(Encoder, self).__init__()
        self.base_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.pblstm = pBLSTM(hidden_dim * 2, hidden_dim)

        self.key_network = nn.Linear(hidden_dim * 2, key_size)
        self.value_network = nn.Linear(hidden_dim * 2, value_size)

    def forward(self, x, lens):
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)

        packed_out, _ = self.base_lstm(rnn_inp)

        # outputs (max_seq_len / 4, batch_size, hidden_size * 2)
        packed_out = self.pblstm(packed_out)

        # outputs (max_seq_len / 4, batch_size, hidden_size * 2)
        # len_out (batch_size)
        # For tests: outputs (130, 64, 256), len_out [55, 58, 66, 60, 62, ...]
        linear_input, lens_out = utils.rnn.pad_packed_sequence(packed_out)

        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value, lens_out


class Decoder(nn.Module):
    """
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step,
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    """
    def __init__(self, vocab_size, hidden_dim, value_size, key_size, is_attended):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.is_attended = is_attended
        if is_attended:
            self.attention = Attention()

        self.teacher_forcing_rate = 0.6

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

    def forward(self, key, values, lens, text=None, is_train=True):
        """
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param lens: (N) lens for key and values
        :param text: (N, text_len) Batch input of text with text_length
        :param is_train: Train or eval mode
        :return predictions: Returns the character prediction probability
        """
        batch_size = key.shape[1]
        embeddings = None

        if is_train:
            max_len = text.shape[1]
            # embeddings (batch_size, text_len, hidden_size)
            embeddings = self.embedding(text)
        else:
            max_len = 250

        predictions = []
        hidden_states = [None, None]
        prediction = (torch.ones(batch_size, 1) * SOS_INDEX).to(DEVICE)

        attention_score = values.mean(dim=0)

        for i in range(max_len):
            # * Implement Gumble noise and teacher forcing techniques 
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break 
            #   out of the loop so you do you do not get index out of range errors. 

            if is_train:

                rnd = np.random.rand()

                if rnd >= self.teacher_forcing_rate:
                    char_embed = embeddings[:, i, :]
                else:
                    char_embed = self.embedding(prediction.argmax(dim=-1))
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            inp = torch.cat([char_embed, attention_score], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            # query
            output = hidden_states[1][0]

            if self.is_attended:
                attention_score, attention_mask = self.attention(output, key, values, lens)

            prediction = self.character_prob(torch.cat([output, attention_score], dim=1))
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)


class Seq2Seq(nn.Module):
    """
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    """
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size, key_size, is_attended=True):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, value_size, key_size)
        self.decoder = Decoder(vocab_size, hidden_dim, value_size, key_size, is_attended)

    def forward(self, speech_input, speech_len, text_input=None, is_train=True):
        key, value, lens = self.encoder(speech_input, speech_len)

        if is_train:
            predictions = self.decoder(key, value, lens, text_input)
        else:
            predictions = self.decoder(key, value, lens, text=None, is_train=False)

        return predictions


def greedy_search_gen(outputs):
    decoded_outputs = torch.argmax(outputs, dim=2)

    decoded_outputs = torch.cat([torch.ones((decoded_outputs.shape[0], 1), dtype=torch.int64).to(DEVICE) * SOS_INDEX,
                                 decoded_outputs], dim=1)

    cur_text_len = torch.zeros(decoded_outputs.shape[0], dtype=torch.int64).to(DEVICE)
    cur_text = []

    for i in range(decoded_outputs.shape[0]):
        cur_text_len[i] = next(j for j in range(decoded_outputs.shape[1])
                               if (decoded_outputs[i][j] == EOS_INDEX or j == decoded_outputs.shape[1] - 1)) + 1

        cur_text.append(decoded_outputs[i][:cur_text_len[i]])

    return cur_text, cur_text_len
