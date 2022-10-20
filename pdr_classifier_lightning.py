from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models

from data_loader.infrared_loader import (DatasetIR, aug_transform_pipeline,
                                         transform_pipeline)

class ImagenetTransferLearning(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)

        num_target_classes = 1
        self.classifier = torch.nn.Linear(num_filters, num_target_classes)

        self.auroc_train = torchmetrics.classification.BinaryAUROC(validate_args=True)
        self.auroc_val = torchmetrics.classification.BinaryAUROC(validate_args=True)
        self.auroc_test = torchmetrics.classification.BinaryAUROC(validate_args=True)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

# class PDRClassifier(pl.LightningModule):

#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.auroc_train = torchmetrics.classification.BinaryAUROC(validate_args=True)
#         self.auroc_val = torchmetrics.classification.BinaryAUROC(validate_args=True)
#         self.auroc_test = torchmetrics.classification.BinaryAUROC(validate_args=True)

    def training_step(self, batch, batch_idx):

        x, y = batch

        # y = torch.unsqueeze(y, -1)
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("train_loss", loss)

        y_hat = logits.sigmoid()
        batch_auroc = self.auroc_train(y_hat, y.int())
        self.log("train_auroc", batch_auroc)

        return loss

    def training_epoch_end(self, outputs):
        self.auroc_train.reset()

    def validation_step(self, batch, batch_idx):

        x, y = batch

        # y = torch.unsqueeze(y, -1)
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("val_loss", loss)

        y_hat = logits.sigmoid()
        self.auroc_val.update(y_hat, y.int())
        
        return loss

    def validation_epoch_end(self, outputs):
        self.log("val_auroc", self.auroc_val.compute())
        self.auroc_val.reset()

    def test_step(self, batch, batch_idx):

        x, y = batch

        # y = torch.unsqueeze(y, -1)
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("test_loss", loss)

        y_hat = logits.sigmoid()
        self.auroc_test.update(y_hat, y.int())

        return loss

    def test_epoch_end(self, outputs):
        self.log("test_auroc", self.auroc_test.compute())
        self.auroc_test.reset()

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.00001, momentum=0.9)
        return optimizer


if __name__ == "__main__":

    # model_instance = getattr(resnet, "resnet50")(pretrained=True, num_classes=1)
    classifier = ImagenetTransferLearning()

    train_loader = torch.utils.data.DataLoader(
        DatasetIR(Path("/storage/homefs/ds21n601/diabetic_retinopathy_classification/dataset/train.csv"), aug_transform_pipeline),
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        DatasetIR(Path("/storage/homefs/ds21n601/diabetic_retinopathy_classification/dataset/validation.csv"), transform_pipeline),
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        DatasetIR(Path("/storage/homefs/ds21n601/diabetic_retinopathy_classification/dataset/test.csv"), transform_pipeline),
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1, num_nodes=1, log_every_n_steps=1)#, strategy="ddp")
    trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader)
