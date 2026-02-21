import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        self._optim.zero_grad()
        
        # -propagate through the network
        outputs = self._model(x)
        
        # -calculate the loss
        loss = self._crit(outputs, y)
        
        # -compute gradient by backward propagation
        loss.backward()
        
        # -update weights
        self._optim.step()
        
        # -return the loss
        return loss.item()
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        outputs = self._model(x)
        loss = self._crit(outputs, y)
        
        # Convert outputs to predictions (threshold at 0.5)
        predictions = (outputs > 0.5).float()
        
        # return the loss and the predictions
        return loss.item(), predictions
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        # set training mode
        self._model.train()
        
        # iterate through the training set
        epoch_loss = 0.0
        for x, y in tqdm(self._train_dl, desc='Training', leave=False):
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            
            # perform a training step
            loss = self.train_step(x, y)
            epoch_loss += loss
        
        # calculate the average loss for the epoch and return it
        avg_loss = epoch_loss / len(self._train_dl)
        return avg_loss
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        self._model.eval()
        
        # disable gradient computation
        with t.no_grad():
            epoch_loss = 0.0
            all_predictions = []
            all_labels = []
            
            # iterate through the validation set
            for x, y in tqdm(self._val_test_dl, desc='Validation', leave=False):
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                
                # perform a validation step
                loss, predictions = self.val_test_step(x, y)
                epoch_loss += loss
                
                # save the predictions and the labels for each batch
                all_predictions.append(predictions.cpu())
                all_labels.append(y.cpu())
            
            # calculate the average loss
            avg_loss = epoch_loss / len(self._val_test_dl)
            
            # calculate metrics
            all_predictions = t.cat(all_predictions, dim=0).numpy()
            all_labels = t.cat(all_labels, dim=0).numpy()
            
            # Calculate F1 score (mean across both classes)
            f1 = f1_score(all_labels, all_predictions, average='samples', zero_division=0)
            
            # print the calculated metrics
            print(f'Val Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}')
            
            # return the loss
            return avg_loss
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        epoch_counter = 0
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        while True:
      
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            # stop by epoch number
            if epochs > 0 and epoch_counter >= epochs:
                break
            
            # train for an epoch
            print(f'\nEpoch {epoch_counter + 1}/{epochs if epochs > 0 else "∞"}')
            train_loss = self.train_epoch()
            
            # calculate the loss and metrics on the validation set
            val_loss = self.val_test()
            
            # append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Train Loss: {train_loss:.4f}')
            
            # use the save_checkpoint function to save the model
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch_counter)
                epochs_no_improve = 0
                print(f'Validation loss improved! Checkpoint saved.')
            else:
                epochs_no_improve += 1
                print(f'No improvement for {epochs_no_improve} epoch(s)')
            
            # check whether early stopping should be performed
            if self._early_stopping_patience > 0:
                if epochs_no_improve >= self._early_stopping_patience:
                    print(f'\nEarly stopping triggered after {epochs_no_improve} epochs without improvement')
                    break
            
            epoch_counter += 1
        
        # return the losses for both training and validation
        return train_losses, val_losses
                    
        
        
        
