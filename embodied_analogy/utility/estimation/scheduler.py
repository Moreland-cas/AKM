import torch
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
        初始化早停与学习率调度器结合的类

        参数:
            optimizer: 优化器
            factor: 学习率衰减因子
            patience: int, 验证指标在多少个 epoch 内没有改善时降低学习率
            early_stop_patience: int, 验证指标在多少个 epoch 内没有改善时触发早停
            delta: float, 验证指标改善的最小变化量
            path: str, 保存最佳模型的路径
        """
        self.optimizer = optimizer
        self.factor = lr_update_factor
        self.patience = lr_scheduler_patience
        self.early_stop_patience = early_stop_patience
        
        # 初始化学习率调度器
        self.scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=self.factor, patience=self.patience, verbose=True)

        # 早停相关变量
        self.best_loss = 1e9
        self.best_state_dict = {}
        
        # 代表有多久没有下降了
        self.counter = 0

    def step(self, cur_loss, state_dict):
        """
        在每次用 cur_state_dictc 计算出 cur_loss 后调用本函数, 更新学习率并检查早停条件

        参数:
            cur_loss: float, 当前计算出的待优化损失
            state_dict: 当前的待优化参数状态
        """
        # 更新学习率调度器
        self.scheduler.step(cur_loss)

        # 检查早停条件
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
