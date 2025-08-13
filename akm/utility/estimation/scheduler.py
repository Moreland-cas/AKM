from torch.optim.lr_scheduler import ReduceLROnPlateau


class Scheduler:
    def __init__(
        self,
        optimizer,
        lr_update_factor=0.5,
        lr_scheduler_patience=3,
        early_stop_patience=10,
    ):
        """
        Initializes a class that combines early stopping with a learning rate scheduler.

        Parameters:
        optimizer: optimizer
        factor: learning rate decay factor
        patience: int, number of epochs after which the validation metric shows no improvement before reducing the learning rate
        early_stop_patience: int, number of epochs after which the validation metric shows no improvement before triggering early stopping
        delta: float, minimum delta required to improve the validation metric
        path: str, path to the best model
        """
        self.optimizer = optimizer
        self.factor = lr_update_factor
        self.patience = lr_scheduler_patience
        self.early_stop_patience = early_stop_patience
        
        self.scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=self.factor, patience=self.patience, verbose=True)
        self.best_loss = 1e9
        self.best_state_dict = {}
        # How long has it been since it last dropped?
        self.counter = 0

    def step(self, cur_loss, state_dict):
        """
        This function is called each time cur_loss is calculated using cur_state_dictc to update the learning rate and check the early stopping condition.

        Parameters:
            cur_loss: float, the currently calculated loss to be optimized
            state_dict: the current parameter state to be optimized
        """
        self.scheduler.step(cur_loss)
        if cur_loss < self.best_loss:
            self.counter = 0
            self.best_loss = cur_loss
            self.best_state_dict = state_dict
            
        else:
            self.counter += 1
            if self.counter >= self.early_stop_patience:
                return True
            else:
                return False