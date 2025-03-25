from bolts.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, resnet18
from lightning import LightningModule, Trainer
import numpy as np
import torchmetrics
import sys, os
# sys.path.append(os.path.abspath("/baselinescvpr"))

# from bcl.BalancedContrastiveLearning.models import resnext
# from sbcl.SBCL.SimCLR.resnet import SupConResNet


class LinearClassifier(LightningModule):
    def __init__(self,
                 input_size: int = 2048,
                 num_classes: int = 2,
                 p_dropout: float = 0.0,
                 ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(input_size, num_classes, bias=True)
        )

    def forward(self, x) -> torch.Tensor:
        return self.classifier(x)


class MLPClassifier(LightningModule):
    def __init__(self,
                 input_size: int = 2048,
                 hidden_dim_size: int = 2048,
                 num_classes: int = 2,
                 p_dropout: float = 0.0,
                 ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_dim_size, bias=False),
            nn.BatchNorm1d(hidden_dim_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(hidden_dim_size, num_classes, bias=True),
        )
    def forward(self, x) -> torch.Tensor:
        return self.classifier(x)


def load_bcl_backbone(checkpoint):
    # model  = resnext.BCLModel(name='resnet50', num_classes=2, feat_dim=128,
    #                              use_norm=True)
    # # remove prefix module
    # new_state_dict = {}
    # for k, v in checkpoint['state_dict'].items():
    #     if k.startswith('module.'):
    #         k = k[7:]  # Remove the first 7 characters ('module.')
    #     new_state_dict[k] = v

    # # Load the modified state_dict into the model
    # model.load_state_dict(new_state_dict)
    # return model.encoder
    
    pass


def load_sbcl_backbone(checkpoint):
    # model = SupConResNet(feat_dim=128)
    # model.load_state_dict(checkpoint)
    # return model.encoder
    pass



class FineTuneClassifier(LightningModule):
    def __init__(self,
                 base_model: LightningModule = None,
                 num_ftrs: int = 2048,
                 num_classes: int = 2,
                 max_epochs: int = 100,
                 lr: float = 1e-3,
                 nesterov: bool = False,
                 p_dropout: float = 0.0,
                 weight_decay: float = 0,
                 hidden_dim_size = None,
                 warmup_epochs: int = 0,
                 optimizer_name: str = "adam",
                 trainable_encoder: bool = False,
                 use_backbone: bool = True,
                 base_model_path = None,
                 all4one_workaround = False,
                 load_special_encoder = None,
                 use_neurips_fix=False
                 ):
        super().__init__()

        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.p_dropout = p_dropout
        self.nesterov = nesterov
        self.warmup_epochs = warmup_epochs
        self.optimizer_name = optimizer_name
        self.trainable_encoder = trainable_encoder
        self.use_backbone = use_backbone
        self.all4one_workaround = all4one_workaround
        self.use_neurips_fix = use_neurips_fix
        
        if self.use_neurips_fix:
            prototypes_np = np.array([[1.,  0.],
                                      [0., -1.]], dtype=np.float32)
            self.register_buffer(
                "prototypes_matrix",
                torch.from_numpy(prototypes_np)
            )

        self.base_model = base_model

        if load_special_encoder is not None:
            print("hjdfhs")

            if load_special_encoder == "bcl":
                print("Loading BCL")
                self.base_model = load_bcl_backbone(torch.load(base_model_path))
            elif load_special_encoder == "sbcl":
                self.base_model = load_sbcl_backbone(torch.load(base_model_path))
            else:
                raise NotImplementedError
        else:

            if base_model_path is not None:
                self.base_model.load_state_dict(torch.load(base_model_path)['state_dict'])

        

        print(f"inside {type(self.base_model)}")
       


        if hidden_dim_size is None:
            print("Using LinearClassifier for finetuning")
            self.classifier = LinearClassifier(input_size=num_ftrs, num_classes=num_classes, p_dropout=p_dropout)
        else:
            print("Using MLP for finetuning")
            self.classifier = MLPClassifier(input_size=num_ftrs, hidden_dim_size=hidden_dim_size, num_classes=num_classes, p_dropout=p_dropout)



   
        
        self.val_auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_classes)
        self.test_auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_classes)

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

        self.test_binf1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()

        if not self.trainable_encoder:

            
            if self.use_backbone:
                for param in self.base_model.parameters():
                    param.requires_grad = False
                self.base_model.eval()
              
            else:
                self.base_model.eval()
           
                if self.all4one_workaround:
                    print("Freezing the encoder all4one")
                    learnable_params = self.base_model.learnable_params
                    print(learnable_params)
        
                    # Freeze each parameter by setting requires_grad to False
                    for param_group in learnable_params:
                        for param in param_group["params"]:
                            param.requires_grad = False

        else:
            self.base_model.train()
            print("Training the encoder")


        


    def forward(self, x) -> torch.Tensor:
        
        if not self.trainable_encoder:
            with torch.no_grad():
                if self.use_backbone:
                    out = self.base_model(x)
                else:
                    out = self.base_model(x)
                if type(out) == dict: #all4one
                    y_hat = out['feats']
                else:
                    if len(out) == 2:
                        y_hat, _ = out
                    else:
                        y_hat = out
        else:
            if self.use_backbone:
                out = self.base_model.backbone(x)
            else:
                out = self.base_model(x)
            if len(out) == 2:
                y_hat, _ = out
            else:
                y_hat = out
            # y_hat, _ = self.base_model(x)

        y_hat = y_hat.view(y_hat.size(0), -1)  # Flatten the output

        return self.classifier(y_hat) #disconnect a tensor from the computation graph and stop further operations from being tracked


    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch

        if len(y.shape)>1:
            y = y.flatten()

        preds = self.forward(x)
        
        print(preds.shape)
        
        if self.use_neurips_fix:
            preds = torch.mm(preds, self.prototypes_matrix)

        loss = self.criterion(preds, y)
        self.log('train.loss.epoch', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,)
        self.log('train.loss', loss,)
        self.train_acc(preds.softmax(-1), y)
        self.log('train.acc', self.train_acc,on_epoch=True,)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if len(y.shape)>1:
            y = y.flatten()

        preds = self.forward(x)
        if self.use_neurips_fix:
            preds = torch.mm(preds, self.prototypes_matrix)

        loss = self.criterion(preds, y)
        self.log('val.loss', loss, on_epoch=True)

        probs = preds.softmax(-1)
        self.val_acc(probs, y)
        self.val_auroc(probs, y)

        self.log('val.acc', self.val_acc)
        self.log('val.auroc', self.val_auroc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        if len(y.shape)>1:
            y = y.flatten()

        preds = self.forward(x)
        if self.use_neurips_fix:
            preds = torch.mm(preds, self.prototypes_matrix)

        loss = self.criterion(preds, y)
        self.log('test.loss', loss)

        probs = preds.softmax(-1)
        self.test_acc(probs, y)
        self.test_auroc(probs, y)
        self.test_binf1(probs, y)

        self.log('test.acc', self.test_acc)
        self.log('test.auc', self.test_auroc)
        self.log('test.binf1', self.test_binf1)

    def configure_optimizers(self):
        lr_decay_rate = 0.1

        if self.optimizer_name == "adam":
             optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
             print("Using Adam optimizer")
        else:

            optimizer = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.lr,
                nesterov=self.nesterov,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.1, last_epoch=-1
            )
        if self.warmup_epochs > 0:
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                self.warmup_epochs,
                self.max_epochs,
                warmup_start_lr = self.lr * 0.1,
                eta_min = self.lr * (lr_decay_rate ** 3),
                last_epoch = -1,
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=self.lr * (lr_decay_rate ** 3), last_epoch=-1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler":lr_scheduler,
                "interval": "epoch",
                "frequency": 1,

            },}
    
