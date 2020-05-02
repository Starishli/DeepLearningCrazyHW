import time
import torch
### Add Your Other Necessary Imports Here! ###

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    model.to(DEVICE)

    criterion = criterion.to(DEVICE)

    start = time.time()

    # 1) Iterate through your loader
    for cur_speech, cur_speech_len, cur_text, cur_text_len in train_loader:

        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        # 3) Set the inputs to the device.
        # 4) Pass your inputs, and length of speech into the model.
        # outputs = model(cur_speech, cur_speech_len, cur_text)
        # 5) Generate a mask based on the lengths of the text to create a masked loss.
        # 5.1) Ensure the mask is on the device and is the correct shape.
        # 6) If necessary, reshape your predictions and origianl text input
        # 6.1) Use .contiguous() if you need to.
        # 7) Use the criterion to get the loss.
        # 8) Use the mask to calculate a masked loss.
        # 9) Run the backward pass on the masked loss.
        # 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        # 11) Take a step with your optimizer
        # 12) Normalize the masked loss
        # 13) Optionally print the training loss after every N batches

        torch.autograd.set_detect_anomaly(True)

        cur_speech = cur_speech.to(DEVICE)  # (max_seq_len, batch_size, utter_len (or 40))
        cur_speech_len = cur_speech_len.to(DEVICE)  # (batch_size)
        cur_text = cur_text.to(DEVICE)  # (batch_size, max_seq_len)
        cur_text_len = cur_text_len.to(DEVICE)  # (batch_size)

        outputs = model(cur_speech, cur_speech_len, cur_text)

        outputs_mask = torch.arange(cur_text.shape[1]).reshape((-1, 1)).to(DEVICE) < cur_text_len
        optimizer.zero_grad()

        loss = None
        n_tokens = cur_text_len.sum()

        for i in range(cur_text.size(1) - 1):
            cur_output = outputs[:, i, :]
            active = outputs_mask[i, :]

            if loss is None:
                loss = criterion(cur_output[active], cur_text[active, i + 1])
            else:
                loss += criterion(cur_output[active], cur_text[active, i + 1])

        loss /= n_tokens
        loss.backward()
        optimizer.step()

        print(loss)

    end = time.time()


def test(model, test_loader, epoch):
    ### Write your test code here! ###
    pass