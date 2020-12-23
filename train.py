from aim.pytorch_lightning import AimLogger
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import IPython
import math
from pathlib import Path
import os


class PySourceDataset(Dataset):
    def __init__(self, main_folder, seq_len=80):
        if not os.path.exists(main_folder):
            raise ValueError("This folder doesn't exist")

        self.main_folder = main_folder
        self.seq_len = seq_len

        # Read all python source files
        files = list()
        for path in Path(main_folder).rglob("*.py"):
            with open(path, encoding="utf8") as fp:
                files.append(fp.read())
        print("Number of .py files:", len(files))

        #  Concatenate them with a page separator
        #  corpus = "\n\nNEW FILE\n\n".join(files)
        self.corpus = "\n".join(files)
        self.corpus_len = len(self.corpus)
        print("Chars in corpus:", self.corpus_len)

        # Define useful mappings
        idx = 0
        self.c2i = dict()
        for c in self.corpus:
            if c not in self.c2i:
                self.c2i[c] = idx
                idx += 1
        self.i2c = {v: k for k, v in self.c2i.items()}
        self.n_chars = len(self.c2i)
        print("Number of distinct chars:", self.n_chars)

    def _get_onehot(self, c):
        t = torch.zeros(1, self.n_chars)
        t[0][self.c2i[c]] = 1
        return t

    def __len__(self):
        return self.corpus_len - self.seq_len

    def __getitem__(self, idx):
        assert idx < len(self)

        #  raw text sequences
        source_seq = self.corpus[idx : idx + self.seq_len]
        target_seq = self.corpus[idx + 1 : idx + self.seq_len + 1]

        #  one-hot
        source_seq_t = torch.stack([self._get_onehot(c) for c in source_seq]).squeeze(dim=1)
        target_seq_t = torch.Tensor([self.c2i[c] for c in target_seq]).long()

        return (source_seq_t, target_seq_t)


class CharLM(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        learning_rate,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.Who = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x, hidden_state=None):
        out, hidden_state = self.lstm(x, hidden_state)
        out = self.Who(out)
        logprob = self.softmax(out)
        return logprob, hidden_state

    def training_step(self, batch, batch_idx):
        source, target = batch
        log_prob, (state_h, state_c) = self(source)

        # store the last step's hidden state
        # TODO should this be fed as initial state to the next sequence?
        # self.hidden_state = (state_h.detach(), state_c.detach())

        # compute sum of losses across time steps
        loss = F.nll_loss(log_prob.view(-1, log_prob.shape[2]), target.view(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("source_folder")
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--n_layers", default=3, type=int)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5) 
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default="pl_charLM")
    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------

    py_train_dataset = PySourceDataset(args.source_folder)   # add here a bunch of repositories
    train_loader = DataLoader(
        py_train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------
    # model
    # ------------
    input_size = output_size = py_train_dataset.n_chars
    model = CharLM(
        input_size,
        args.hidden_size,
        output_size,
        learning_rate=args.learning_rate,
        num_layers=args.n_layers,
        dropout=args.dropout
    )

    # ------------
    # training
    # ------------
    aim_logger = AimLogger(
        experiment=args.exp_name,
        train_metric_prefix="train_",
        test_metric_prefix="test_",
        val_metric_prefix="val_",
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = aim_logger
    trainer.fit(model, train_loader)

    # ------------
    # testing
    # ------------
    #  trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
