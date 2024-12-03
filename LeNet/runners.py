import torch as nn


def train(model, epochs=int, train_dataloader, optimizer, loss_fn, verbose=True):
    model.train()
    losses = []
    for images, label in tqdm(train_dataloader):

        optimizer.zero_grad()
        pred = model(images)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        losses.append(loss)
    
    if verbose:
        print(f'Epoch {epoch} | Loss: {loss}')

    return model, losses

def test(model, test_dataloader):
    model.eval()