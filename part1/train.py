import argparse
import torch
import numpy as np
import os
import pickle

from torch.utils.data import Subset, DataLoader
import torch.nn as nn

from model import FFNN
from loader import POSTaggedDataset
from tqdm import tqdm

from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
import datetime
from torchmetrics import Precision, Recall, F1Score, FBetaScore
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassFBetaScore
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

# create a timestamp to use in the checkpoint directory name


def plot_and_save_confusion_matrix(writer, run_path, index2tag, target_labels, pred_labels, labels):
    target_labels = list(map(lambda x: index2tag[x], target_labels.tolist()))
    pred_labels = list(map(lambda x: index2tag[x], pred_labels.tolist()))
    confusion_matrix = metrics.confusion_matrix(
        target_labels, pred_labels, labels=labels)
    cmdf = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    cmdf.to_csv(f'{run_path}confusion_matrix.csv')
    plt.figure(figsize=(16, 12))
    figure = sns.heatmap(cmdf, annot=True, fmt='.0f', cmap=sns.color_palette(
        "viridis", as_cmap=True)).get_figure()
    figure.savefig(f'{run_path}confusion_matrix.png')
    writer.add_figure('kfold_confusion_matrix_sum',
                      figure)


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser()

    model_args = parser.add_argument_group('model')
    model_args.add_argument('--input_dim', type=int,
                            default=300, help='Number of input dimensions')
    model_args.add_argument('--output_dim', type=int,
                            default=12, help='Number of output dimensions')
    model_args.add_argument('--hidden_dim', type=int, action='append',
                            help='Number of hidden dimensions [0th, 1st, 2nd]')
    model_args.add_argument(
        '--lr', type=float, default=1e-3, help="Learning rate")
    model_args.add_argument('--context', type=int, default=2,
                            help='Context size (onesided)')

    training_args = parser.add_argument_group('training')
    training_args.add_argument(
        '--seed', type=int, default=42, help='Random seed for experiments')
    training_args.add_argument('--n_epochs', type=int,
                               default=10, help='Number of training epochs')
    training_args.add_argument('--num_workers', type=int,
                               default=1, help='Number of CPU workers')
    training_args.add_argument('--batch_size', type=int,
                               default=128, help='Batch size for training dataloader')
    training_args.add_argument('--savedir', type=str, default='runs',
                               help='Root directory where all checkpoint and logs will be saved')

    args = parser.parse_args()

    input_dim = args.input_dim
    output_dim = args.output_dim
    hidden_dim = args.hidden_dim
    learning_rate = args.lr
    context_size = args.context

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    savedir = args.savedir
    num_workers = args.num_workers
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    corpus_index = pickle.load(open('../data/corpus_index.pickle', 'rb'))
    dataset = POSTaggedDataset(corpus_index, context_size, 300)

    run_name = f'epochs={n_epochs},batch_size={batch_size},hidden_dim={hidden_dim},timestamp={timestamp}'

    writer = SummaryWriter(f'runs/{run_name}/tf_logs')

    with open(f'runs/{run_name}/hparams.json', 'w') as f:
        f.write(json.dumps(vars(args), indent=2))

    splits = KFold(n_splits=5, shuffle=True, random_state=seed)

    precision = Precision(
        task="multiclass", average='micro', num_classes=12).to(device)
    recall = Recall(task="multiclass", average='micro',
                    num_classes=12).to(device)
    f1 = F1Score(task="multiclass", num_classes=12).to(device)
    fpoint5 = FBetaScore(
        task="multiclass", num_classes=12, beta=0.5).to(device)
    f2 = FBetaScore(task="multiclass", num_classes=12, beta=2.0).to(device)

    multiclass_recall = MulticlassRecall(
        num_classes=12, average=None).to(device)
    multiclass_precision = MulticlassPrecision(
        num_classes=12, average=None).to(device)
    multiclass_f1 = MulticlassF1Score(num_classes=12, average=None).to(device)
    multiclass_fpoint5 = MulticlassFBetaScore(
        num_classes=12, average=None, beta=0.5).to(device)
    multiclass_f2 = MulticlassFBetaScore(
        num_classes=12, average=None, beta=2.0).to(device)

    target_labels = torch.tensor([]).to(device)
    pred_labels = torch.tensor([]).to(device)

    index2tag = pickle.load(open('../data/index_to_tag.pickle', 'rb'))
    labels = list(index2tag.values())

    # avg_precision, avg_recall, avg_fpoint5, avg_f1, avg_f2 = 0, 0, 0, 0, 0

    for fold, (train_index, val_index) in enumerate(splits.split(np.arange(len(dataset)))):

        model = FFNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)

        train_dataloader = DataLoader(
            train_subset, batch_size=batch_size, num_workers=num_workers
        )

        val_dataloader = DataLoader(
            val_subset, batch_size=batch_size, num_workers=num_workers
        )

        training_loss = 0.0
        total_steps = len(train_dataloader)

        for epoch in range(n_epochs):
            for i, batch in enumerate(tqdm(train_dataloader, desc=f'fold={fold}, epoch={epoch} progress')):
                X, t = batch  # sir's notation t: true label; y: observed label
                X = X.type(torch.float32)
                t = t.type(torch.float32)
                X, t = X.to(device), t.to(device)

                optimizer.zero_grad()

                y = model.forward(X)
                loss = loss_fn(y, t)

                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                if (i+1) % 100 == 0:
                    writer.add_scalar(
                        f'fold_{fold}/training_loss', training_loss / 100, epoch * total_steps + i)
                    training_loss = 0.0

        model.eval()
        val_loss = 0
        total_steps = len(val_dataloader)

        pred = torch.tensor([]).to(device)
        target = torch.tensor([]).to(device)

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader, desc=f'fold={fold}, validate progress:')):
                X, t = batch
                X = X.type(torch.float32)
                t = t.type(torch.float32)
                X, t = X.to(device), t.to(device)
                y = model.forward(X)
                loss = loss_fn(y, t)
                val_loss += loss.item()
                # self.log("validation_loss", val_loss)
                target = torch.cat((target, torch.argmax(t, dim=1)))
                pred = torch.cat((pred, torch.argmax(y, dim=1)))

        target_labels = torch.cat((target_labels, target))
        pred_labels = torch.cat((pred_labels, pred))

        precision_score = precision(pred, target)
        recall_score = recall(pred, target)
        fpoint5_score = fpoint5(pred, target)
        f1_score = f1(pred, target)
        f2_score = f2(pred, target)

        writer.add_scalar(
            f'precision', precision_score, fold)
        writer.add_scalar(
            f'recall', recall_score, fold)
        writer.add_scalar(f'f_0.5',
                          fpoint5_score, fold)
        writer.add_scalar(f'f_1',
                          f1_score, fold)
        writer.add_scalar(f'f_2',
                          f2_score, fold)

        # avg_precision = (avg_precision * fold + precision_score) / (fold + 1)
        # avg_recall = (avg_recall * fold + recall_score) / (fold + 1)
        # avg_fpoint5 = (avg_fpoint5 * fold + fpoint5_score) / (fold + 1)
        # avg_f1 = (avg_f1 * fold + f1_score) / (fold + 1)
        # avg_f2 = (avg_f2 * fold + f2_score) / (fold + 1)

        if fold == 0:
            os.makedirs(f'runs/{run_name}/eval/', exist_ok=True)
            os.makedirs(f'runs/{run_name}/folds_model/', exist_ok=True)
            os.makedirs(f'runs/{run_name}/outputs/', exist_ok=True)
        with open(f'runs/{run_name}/eval/eval_metrics_{fold}.json', 'w') as f:
            f.write(json.dumps({
                'precision': precision_score.item(),
                'recall': recall_score.item(),
                'f0.5': fpoint5_score.item(),
                'f1': f1_score.item(),
                'f2': f2_score.item(),
                'tag_wise': {
                    'precision': multiclass_precision(pred, target).tolist(),
                    'recall': multiclass_recall(pred, target).tolist(),
                    'f1': multiclass_f1(pred, target).tolist(),
                    'f0.5': multiclass_fpoint5(pred, target).tolist(),
                    'f2': multiclass_f2(pred, target).tolist(),
                }
            }, indent=2))
        pickle.dump({'prediction': pred.to('cpu'), 'target': target.to('cpu')}, open(
            f'runs/{run_name}/outputs/op_fold_{fold}.pickle', 'wb'))

        torch.save(model.state_dict(),
                   f'runs/{run_name}/folds_model/model_{fold}.pt')
    plot_and_save_confusion_matrix(
        writer, f'runs/{run_name}/', index2tag, target_labels, pred_labels, labels=labels)

    writer.close()

    model = FFNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )

    training_loss = 0.0
    total_steps = len(train_dataloader)
    for epoch in range(n_epochs):
        for i, batch in enumerate(tqdm(train_dataloader, desc=f'Final Model, epoch={epoch} progress')):
            X, t = batch  # sir's notation t: true label; y: observed label
            X = X.type(torch.float32)
            t = t.type(torch.float32)
            X, t = X.to(device), t.to(device)

            optimizer.zero_grad()

            y = model.forward(X)
            loss = loss_fn(y, t)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            if (i+1) % 100 == 0:
                writer.add_scalar(
                    f'final/training_loss', training_loss / 100, epoch * total_steps + i)
                training_loss = 0.0

    torch.save(model.state_dict(), f'runs/{run_name}/final_model.pt')


if __name__ == '__main__':
    main()
