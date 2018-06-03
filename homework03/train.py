import torch
from torch.autograd import Variable
from os.path import join
import shutil
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm


def variable(x, volatile=False):
    return Variable(x, volatile=volatile).cuda()


def get_float(v):
    result = v
    if isinstance(v, torch.autograd.Variable):
        result = v.data[0]
    return result


def train(
    model,
    get_loss,
    output_dirpath,
    init_optimizer, lr,
    epochs,
    criterion,
    train_dataloader, val_dataloader,
    val_freq,
    patience=5
):
    model.train()
    optimizer = init_optimizer(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    next_val = val_freq
    last_lr_reset_val_it = 0
    val_it = 0
    val_losses = []
    train_hist = defaultdict(list)
    for epoch in range(epochs):
        tq = tqdm(total=len(train_dataloader))
        tq.set_description('Epoch {}'.format(epoch + 1))
        tq.refresh()

        it = 0
        its = len(train_dataloader)

        for batch in train_dataloader:
            loss = get_loss(batch, model, criterion, volatile=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = get_float(loss)
            postfix = {'loss': '{:.3f}'.format(loss)}
            tq.set_postfix(**postfix)
            tq.update(1)

            it += 1
            global_progress = epoch + it / its

            train_hist['train'].append((global_progress, loss))

            if global_progress + 1e-9 >= next_val:
                val_it += 1
                model_dirpath = join(output_dirpath, 'model.pth')
                torch.save(model.state_dict(), model_dirpath)
                while not next_val > global_progress:
                    next_val += val_freq
                val_loss = evaluate(model, val_dataloader, get_loss, criterion)
                model.train()
                train_hist['val'].append((global_progress, val_loss))
                val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_dirpath = join(output_dirpath, 'model_best.pth')
                    shutil.copy(model_dirpath, best_model_dirpath)
                else:
                    if (patience and val_it - last_lr_reset_val_it > patience and
                                min(val_losses[-patience:]) > best_val_loss):
                        lr /= 10
                        print('lr dropped to {}'.format(lr))
                        last_lr_reset_val_it = val_it
                        optimizer = init_optimizer(model.parameters(), lr=lr)
        tq.close()
    return train_hist


def evaluate(model, dataloader, get_loss, criterion):
    model.eval()

    tq = tqdm(total=len(dataloader))
    tq.set_description('Eval')
    tq.refresh()

    samples_count = 0
    val_loss = 0
    for batch in dataloader:
        loss = get_loss(batch, model, criterion, volatile=True)
        batch_size = batch[0].shape[0]
        val_loss += get_float(loss) * batch_size
        samples_count += batch_size
        tq.update(1)

    val_loss /= samples_count

    postfix = {'loss': '{:.3f}'.format(val_loss)}
    tq.set_postfix(**postfix)

    tq.close()

    return val_loss