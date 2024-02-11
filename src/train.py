import torch
from earlystopping import EarlyStopping


class CustomTrain:
    def __init__(self,
                 model,
                 train_dataloader,
                 dev_dataloader,
                 num_epochs,
                 save_model_file_name,
                 criterion,
                 optimizer,
                 writer,
                 logger,
                 device):
        '''
        Description: Custom training class to train transformers/pytorch model.

        Arguments
        ----------
        model: model that is required to train.
        train_dataloader: dataloader to load train data.
        dev_dataloader: dataloader to load dev data.
        num_epochs: (int), number of epochs you want your model to train.
        save_model_file_name: (str), name under which you want your trained model to be saved.
        criterion: loss function.
        optimizer: optimizer to train the model.
        writer: tensorboard summary writer.
        logger: recording training logs into the log files.
        '''
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.num_epochs = num_epochs
        self.save_model_file_name = save_model_file_name
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer
        self.logger = logger
        self.device = device
        self.early_stopping = EarlyStopping(flag='pytorch_model', 
                                            path=save_model_file_name)

    def train(self):
        for epoch in range(self.num_epochs):
            # training loop
            train_loss = 0
            self.model.train()
            for step1, (batch, labels) in enumerate(self.train_dataloader):
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                # pytorch model
                outputs = self.model(**batch)
                # cross entropy loss for classification task
                loss = self.criterion(outputs, labels)
                # backpropagation
                loss.backward()
                # optimizer step
                self.optimizer.step()
                # add train loss for each iteratio in an epoch
                train_loss += loss.item()

            self.model.eval()
            # eval loop
            with torch.no_grad():
                dev_loss = 0
                for step2, (batch,labels) in enumerate(self.dev_dataloader):
                    labels = labels.to(self.device)
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs, labels)
                    # track dev loss for each iteration in an epoch
                    dev_loss += loss.item()

            # calculate epoch wise loss
            train_loss = train_loss / (step1+1)
            dev_loss = dev_loss / (step2+1)

            # write in tensorboard
            if self.writer:
                self.writer.add_scalars('loss',
                                        {'train_loss': train_loss,
                                        'dev_loss': dev_loss},
                                        epoch)
                
            # write training details in log files
            if self.logger:
                message = f'Epoch [{epoch+1}/{self.num_epochs}], train_loss:{train_loss}, dev_loss:{dev_loss}'
                self.logger.info(message)
            else:
                print(message)

            self.early_stopping(dev_loss, self.model)

            # if early stopping criteria is positive, then early stop
            if self.early_stopping.early_stop:
                if self.logger:
                    self.logger.info("Early Stopping")
                break

        torch.save(self.model.state_dict(), self.save_model_file_name)
        return self.model
        