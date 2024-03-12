import torch.nn
import torch

class NaNMSELoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,lossmse):
        y_true = torch.squeeze(y_true)
        y_pred = torch.squeeze(y_pred)
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        # 取平均值为loss
        loss = torch.sqrt(torch.nanmean(lossmse(y_true, y_pred)))
        return loss




    
