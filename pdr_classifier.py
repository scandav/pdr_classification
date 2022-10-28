import copy
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

import utils.utils as u
from data_loader.infrared_loader import (DatasetIR, aug_transform_pipeline,
                                         transform_pipeline)
from models import resnet
from utils.gradcam import GradCamResnet

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

RNDM = 91
torch.manual_seed(RNDM)
DIR_UBELIX = Path(__file__).parent.joinpath("inputs")
DIR_LOCAL = Path("/storage/homefs/ds21n601/oct_biomarker_classification/inputs")
DIR_SLICES = "slices"

BATCH_SIZE = 8
WRITING_PER_EPOCH = 3
EPOCHS = 150
LR = 1E-5
LR_DROP = 0.98

class PDRClassifier:

    def __init__(self) -> None:

        self.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        os.makedirs('weights', exist_ok=True)
        path_str = f"PDR_{self.current_time}"

        self.tb_path = Path(__file__).resolve().parents[0].joinpath("runs", path_str)
        self.device = torch.device('cuda:0')
        self.writer = SummaryWriter(self.tb_path)

    def load_datasets(self):

        train_set = DatasetIR(Path("/storage/homefs/ds21n601/diabetic_retinopathy_classification/dataset/train.csv"), aug_transform_pipeline)
        val_set = DatasetIR(Path("/storage/homefs/ds21n601/diabetic_retinopathy_classification/dataset/validation.csv"), transform_pipeline)
        test_set = DatasetIR(Path("/storage/homefs/ds21n601/diabetic_retinopathy_classification/dataset/test.csv"), transform_pipeline)

        self.writing_freq_train = len(train_set) // (WRITING_PER_EPOCH * BATCH_SIZE)
        self.writing_freq_val = len(val_set) // BATCH_SIZE  # Only once per epoch
        self.writing_freq_test = len(test_set) // BATCH_SIZE  # Only once per epoch

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    def load_model(self):
        model = getattr(resnet, "resnet50")(pretrained=True, num_classes=1)
        self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        self.model.to(self.device)

    def train(self):
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: LR_DROP)

        criterion = nn.BCEWithLogitsLoss()

        best_rocauc = 0.0

        for epoch in range(EPOCHS):

            for phase in ['train', 'test', 'validation']:

                running_loss = 0.0
                running_pred = []
                running_true = []

                if phase == 'train':
                    self.model.train()
                    loader = self.train_loader
                    writing_freq = self.writing_freq_train
                    i_train = 0
                elif phase == 'validation':
                    self.model.eval()
                    loader = self.val_loader
                    writing_freq = self.writing_freq_val
                elif phase == 'test':
                    self.model.eval()
                    loader = self.test_loader
                    writing_freq = self.writing_freq_test

                for i, data in enumerate(loader):

                    inputs, labels, _ = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels) # sigmoid is included in BCEWithLogitsLoss

                        if phase == 'train':
                            i_train = i
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                    running_loss += loss.item()
                    running_pred.append(outputs.sigmoid().detach().cpu().numpy())
                    running_true.append(labels.detach().cpu().numpy())

                    if i % writing_freq == (writing_freq - 1):

                        n_epoch = epoch * len(self.train_loader.dataset) // BATCH_SIZE + i_train + 1
                        epoch_loss = running_loss / (writing_freq * BATCH_SIZE)
                        dict_metrics = u.calculate_metrics_sigmoid(running_true, running_pred)
                        epoch_rocauc = dict_metrics['ROC AUC']
                        print(f'{phase} Loss: {epoch_loss} ROC AUC: {epoch_rocauc}')
                        dict_metrics['Loss'] = epoch_loss
                        u.write_to_tb(self.writer, dict_metrics.keys(), dict_metrics.values(), n_epoch, phase=phase)

                        if phase == 'train':
                            u.write_to_tb(self.writer, ['epoch'], [epoch], n_epoch, phase=phase)

                        if phase == 'validation' and epoch >= 20 and epoch_rocauc > best_rocauc:
                        # if phase == 'validation' and epoch_rocauc > best_rocauc:
                            best_rocauc = epoch_rocauc
                            best_model = copy.deepcopy(self.model)

                        # print out metrics on last epoch
                        if phase in ['validation', 'test'] and epoch == EPOCHS - 1:
                            pass

                        running_pred = []
                        running_true = []
                        running_loss = 0.0

            scheduler.step()
            print(f'Epoch {epoch + 1} finished')
    
        # Save best models and create symlink in working directories
        best_rocauc_model_path = Path(__file__).parents[0].joinpath(
            'weights', f'detector_{self.current_time}_bestROCAUC.pth'
        )
        torch.save(best_model.state_dict(), best_rocauc_model_path)
        (self.tb_path / 'output_best_rocauc').mkdir()
        self.tb_path.joinpath('output_best_rocauc', 'detector_bestROCAUC.pth').symlink_to(best_rocauc_model_path)

        # self.infer(best_model, self.tb_path / 'output_best_rocauc')
        # RUN INFERENCE
        threshold = PDRClassifier.plot_results(best_model, self.val_loader, self.tb_path / 'output_best_rocauc', self.device, dtype='val')
        PDRClassifier.plot_results(best_model, self.test_loader, self.tb_path / 'output_best_rocauc', self.device, dtype='test', threshold=threshold)

    @staticmethod
    def plot_results(model, data_loader, save_dir, device, dtype, threshold=None, gradcam=True):

        save_dir.mkdir(exist_ok=True)

        if gradcam:
            gcam = GradCamResnet(model, save_dir)

        ### VALIDATION
        model.eval()

        preds, trues = [], []

        for data in data_loader:

            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)

            preds.append(outputs.sigmoid().detach().cpu().numpy())
            trues.append(labels.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        if threshold is None:
            threshold = PDRClassifier.compute_threshold(trues, preds)

        if gradcam:

            for data in data_loader:

                inputs, labels, filenames = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(inputs)

                outputs = outputs.sigmoid().detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                for tensor, label, output, fname in zip(inputs, labels, outputs, filenames):
                    true_prolif = "Proliferative" if float(label) == 1.0 else "Non Proliferative"
                    pred_prolif = "Proliferative" if float(output) >= threshold else "Non Proliferative"
                    fig_title=f"{fname}\nTrue: {true_prolif}\nPred: {pred_prolif} ({float(output):.2f})\nThreshold: {threshold:.2f}"
                    gcam.run(tensor, fig_title, fname)

        precision, recall, thresholds = metrics.precision_recall_curve(trues, preds)
        ap = metrics.average_precision_score(trues, preds)

        prediction_int = np.zeros_like(preds)
        prediction_int[preds > threshold] = 1

        recall_th = metrics.recall_score(trues, prediction_int)
        precision_th = metrics.precision_score(trues, prediction_int)

        fig, ax = plt.subplots()
        fig.suptitle(f'Threshold: {threshold:.2f}')
        metrics.ConfusionMatrixDisplay.from_predictions(trues, prediction_int, ax=ax, cmap='Blues')
        fig.tight_layout()
        fig.savefig(f'{save_dir}/conf_matrix_{dtype}.png')

        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        # ax.scatter(recall_th, precision_th, c="g", marker=r'$\clubsuit$')
        # ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.title.set_text(f'AP = {ap:0.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        fig.tight_layout()
        fig.savefig(f'{save_dir}/precision_recall_{dtype}.png')

        fpr, tpr, thresholds = metrics.roc_curve(trues, preds)
        specif, sensit = 1 - fpr, tpr
        auc = metrics.auc(fpr, tpr)

        specif_th = 1 - interp1d(thresholds, fpr)(threshold)
        sensit_th = interp1d(thresholds, tpr)(threshold)

        fig, ax = plt.subplots()
        ax.plot(specif, sensit)
        ax.scatter(specif_th, sensit_th, c="r", marker='D')
        ax.plot([0, 1], [1, 0], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.title.set_text(f'AUC = {auc:0.3f}')
        ax.set_xlabel('Specificity')
        ax.set_ylabel('Sensitivity')
        fig.tight_layout()
        fig.savefig(f'{save_dir}/roc_curve_{dtype}.png')

        with open(f"{save_dir}/threshold_{dtype}.txt", "w") as outfile:
            outfile.write(str(threshold))

        with open(f"{save_dir}/metrics_{dtype}.json", "w") as outfile:
            metric_dict = {
                'AUC': float(auc),
                'AP': float(ap),
                'precision': float(precision_th),
                'recall': float(recall_th),
                'specificity': float(specif_th),
                'sensitivity': float(sensit_th)
            }
            json.dump(metric_dict, outfile)

        return threshold

    @staticmethod
    def compute_threshold(trues, preds):
        fpr, tpr, thresholds = metrics.roc_curve(trues, preds)
        specif, sensit = 1 - fpr, tpr
        function = np.sqrt(specif**2 + sensit**2)
        threshold = thresholds[function.argmax()]
        return threshold


if __name__ == "__main__":

    model = PDRClassifier()
    model.load_datasets()
    model.load_model()
    model.train()
