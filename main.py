import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from models import Seq2Seq
from train_test import train, val, test
from dataloader import load_data, create_dictionaries, transform_index_to_letter, \
    collate_train, collate_test, transform_letter_to_index, Speech2TextDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ', '<sos>', '<eos>']
LETTER2INDEX, INDEX2LETTER = create_dictionaries(LETTER_LIST)


def main():
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=128, value_size=128, key_size=256,
                    is_attended=True)

    # cur_model_num = 6
    # model.load_state_dict(torch.load('model_{}'.format(cur_model_num)))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    n_epochs = 30
    batch_size = 64 if DEVICE == 'cuda' else 1

    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    character_text_train = transform_letter_to_index(transcript_train, LETTER2INDEX)
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER2INDEX)

    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    val_dataset = Speech2TextDataset(speech_valid, character_text_valid)
    test_dataset = Speech2TextDataset(speech_test, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)

    for epoch in range(n_epochs):
        train(model, train_loader, criterion, optimizer, epoch)
        val(model, val_loader, criterion, epoch)

    # test(model, test_loader)

    torch.save(model.state_dict(), 'model_{}'.format(1))

    result_gen(test_loader, 1)


def result_gen(test_loader, model_num):
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=128, value_size=128, key_size=256,
                    is_attended=True)

    model.load_state_dict(torch.load('model_{}'.format(model_num + 1)))
    model.eval()

    model = model.to(DEVICE)

    test_text = test(model, test_loader)

    test_text_str = []

    for cur_text in test_text:
        test_text_str.append(transform_index_to_letter(cur_text, LETTER_LIST))

    res_df = pd.DataFrame(test_text_str)
    res_df.to_csv('result_{}.csv'.format(model_num + 1), index=True, header=False)


if __name__ == '__main__':
    main()
