from collections import namedtuple
import torch
import torch.nn as nn
from torchmetrics import Precision, Recall
import torchmetrics
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
import pytorch_lightning as pl

class HGATr_LIGHT(pl.LightningModule):
    def __init__(
          self,
          model,
          n_classes,
          learning_rate,
          data_class_names,
          weights = None,
          device = "cpu",
        ):

        super(HGATr_LIGHT, self).__init__()

        self.automatic_optimization = False

        self.accuracy = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=n_classes)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

        self.accuracy_overall = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, average='micro')  # Overall accuracy
        self.accuracy_avg = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, average='macro')      # Per-class average accuracy
        self.kappa = torchmetrics.CohenKappa(task="multiclass", num_classes=n_classes)

        self.train_accuracy_overall = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, average='micro')  # Overall accuracy
        self.train_accuracy_avg = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, average='macro')      # Per-class average accuracy
        self.train_kappa = torchmetrics.CohenKappa(task="multiclass", num_classes=n_classes)

        self.test_accuracy_overall = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, average='micro')  # Overall accuracy
        self.test_accuracy_avg = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, average='macro')      # Per-class average accuracy
        self.test_kappa = torchmetrics.CohenKappa(task="multiclass", num_classes=n_classes)
        self.test_per_class_accuracy = MulticlassAccuracy(num_classes=n_classes, average=None)
        self.test_per_class_accuracies = None

        self.conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=n_classes)

        self.precision = Precision(task="multiclass", num_classes=n_classes, average="macro")
        self.recall = Recall(task="multiclass", num_classes=n_classes, average="macro")

        self.train_precision = Precision(task="multiclass", num_classes=n_classes, average="macro")
        self.train_recall = Recall(task="multiclass", num_classes=n_classes, average="macro")

        self.test_results = []

        self.class_names = data_class_names

        self.learning_rate = learning_rate

        if weights is not None:
            self.loss_function = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
        else:
            self.loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.vit = model
        self.output = namedtuple('Output', ['logits'])


    def forward(self, x):
        logits = self.vit(x)

        return self.output(logits=logits)


    def training_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images).logits

        loss = self.loss_function(logits, labels)

        self.manual_backward(loss, retain_graph=True)

        optimizer = self.optimizers()  # Ottieni l'ottimizzatore
        optimizer.step()  # Esegui il passo di ottimizzazione
        optimizer.zero_grad()  # Azzera i gradienti

        # Aggiorna l'accuracy con i nuovi dati
        self.train_accuracy.update(logits, labels)
        self.train_precision.update(logits, labels)
        self.train_recall.update(logits, labels)
        self.train_accuracy_overall.update(logits, labels)
        self.train_accuracy_avg.update(logits, labels)
        self.train_kappa.update(logits, labels)

        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images[0].shape[0])

        return loss


    def on_train_epoch_end(self):
        # Calcola la loss media dell'epoca
        avg_train_loss = self.trainer.callback_metrics["loss"].item()
        train_acc = self.train_accuracy.compute()
        train_precision = self.train_precision.compute()
        train_recall = self.train_recall.compute()
        train_accuracy_overall = self.train_accuracy_overall.compute()
        train_accuracy_avg = self.train_accuracy_avg.compute()
        train_kappa = self.train_kappa.compute()

        self.log("accuracy_overall", train_accuracy_overall, prog_bar=True)
        self.log("accuracy_avg", train_accuracy_avg)
        self.log("kappa", train_kappa)


        # Stampa i risultati alla fine dell'epoca
        print(
            f"[Train] Epoch: {self.current_epoch}  Loss: {avg_train_loss:.4f}  "
            f"Accuracy Overall: {train_accuracy_overall:.4f}  "
            f"Accuracy Avg: {train_accuracy_avg:.4f}  "
            f"Kappa: {train_kappa:.4f}"
        )

        # Reset delle metriche per la prossima epoca
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_accuracy_overall.reset()
        self.train_accuracy_avg.reset()
        self.train_kappa.reset()



    def validation_step(self, batch):
        images, labels = batch

        logits = self(images).logits
        sample_loss = self.loss_function(logits, labels)

        # Update all metrics in the collection
        self.accuracy.update(logits, labels)
        self.conf_matrix.update(logits, labels)
        self.precision.update(logits, labels)
        self.recall.update(logits, labels)
        self.accuracy_overall.update(logits, labels)
        self.accuracy_avg.update(logits, labels)
        self.kappa.update(logits, labels)

        # Log loss per batch
        self.log("val_loss", sample_loss, on_epoch=True, prog_bar=True, batch_size=images[0].shape[0])

        return {"loss": sample_loss}


    def on_validation_epoch_end(self):
        # Compute and log all metrics at the end of the epoch
        avg_val_loss = self.trainer.callback_metrics["val_loss"].item()

        computed_accuracy_overall = self.accuracy_overall.compute()
        computed_accuracy_avg = self.accuracy_avg.compute()
        computed_kappa = self.kappa.compute()

        self.log("val_accuracy_overall", computed_accuracy_overall, prog_bar=True)
        self.log("val_accuracy_avg", computed_accuracy_avg)
        self.log("val_kappa", computed_kappa)


        print(
            f"[Validation] Epoch: {self.current_epoch}  "
            f"Loss: {avg_val_loss:.4f}  "
            f"Accuracy Overall: {computed_accuracy_overall:.4f}  "
            f"Accuracy Avg: {computed_accuracy_avg:.4f}  "
            f"Kappa: {computed_kappa:.4f}"
        )

        # Reset metrics for the next epoch
        self.accuracy.reset()
        self.conf_matrix.reset()
        self.precision.reset()
        self.recall.reset()
        self.accuracy_overall.reset()
        self.accuracy_avg.reset()
        self.kappa.reset()


    def test_step(self, batch):
        images, labels = batch

        logits = self(images).logits

        self.test_accuracy_overall.update(logits, labels)
        self.test_accuracy_avg.update(logits, labels)
        self.test_kappa.update(logits, labels)
        self.test_per_class_accuracy.update(logits, labels)

        return {"test_loss": self.loss_function(logits, labels)}


    def on_test_epoch_end(self):
        test_accuracy_overall = self.test_accuracy_overall.compute()
        test_accuracy_avg = self.test_accuracy_avg.compute()
        test_kappa = self.test_kappa.compute()
        per_class_acc = self.test_per_class_accuracy.compute()

        self.test_per_class_accuracies = per_class_acc.cpu().numpy()

        # Print a summary of top-k accuracy metrics and MRR
        print(
            f"[Test] accurracy {test_accuracy_overall:.4f}; "
            f"accuracy avg {test_accuracy_avg:.4f}; "
            f"kappa {test_kappa:.4f}"
        )

        self.test_results = [test_accuracy_overall.item(), test_accuracy_avg.item(), test_kappa.item()]

        print(f"[Test] Accuracy per classe:")
        for i, acc in enumerate(per_class_acc):
            print(f"  Classe {i} ({self.class_names[i]}): {acc:.4f}")

        # Reset metrics for the next epoch
        self.test_accuracy_overall.reset()
        self.test_accuracy_avg.reset()
        self.test_kappa.reset()
        self.test_per_class_accuracy.reset()


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)