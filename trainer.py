import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    writer = SummaryWriter(flush_secs=10)
    writer_train_index = 0 #variable to store training index accross epochs

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()
        # Train stage
        train_loss, metrics, writer_train_index = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, writer, writer_train_index)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        writer.add_scalar("eval_loss", val_loss, epoch)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)

    writer.close()


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, writer, writer_train_index):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target, index) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        #OFFLINE GEN
        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)

        # # ONLINE GEN
        # loss_inputs = outputs
        # if index is not None:
        #     index = (index,)
        #     loss_inputs += index
        # loss_outputs = loss_fn(*loss_inputs)

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        dataset = train_loader.dataset.get_dataset()
        true_knns = train_loader.dataset.get_KNN()

        if batch_idx % log_interval == 0:
            for metric in metrics:
                metric(outputs, target, loss_outputs, dataset, model, data[0], true_knns, index)
            writer_train_index += log_interval
            writer.add_scalar("train_loss", np.mean(losses), writer_train_index)
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics, writer_train_index


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target, index) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            #OFFLINE GEN
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)

            # # ONLINE GEN
            # loss_inputs = outputs
            # if index is not None:
            #     index = (index,)
            #     loss_inputs += index
            # loss_outputs = loss_fn(*loss_inputs)

            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            dataset = val_loader.dataset.get_dataset()
            true_knns = val_loader.dataset.get_KNN()
            for metric in metrics:
                metric(outputs, target, loss_outputs, dataset, model, data[0], true_knns, index)

    return val_loss, metrics
