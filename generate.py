import pytorch_lightning as pl
from main import CharLM, PySourceDataset
import argparse
import secrets

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--steps", type=int, default=200)
args = parser.parse_args()
model = CharLM.load_from_checkpoint(args.model)

name = f"{secrets.token_hex(2)}.py"

prompt = input("> ")

py_train_dataset = PySourceDataset("./scikit-learn-master")
inputs = [py_train_dataset._get_onehot(c) for c in prompt]

logprob = None
hidden_state = None
for i in inputs[:-1]:
    i = i.unsqueeze(0)
    logprob, hidden_state = model(i, hidden_state)

currc = py_train_dataset.i2c[logprob.argmax(dim=2).item()]
generated = list(prompt) + [currc]  # first item of the generated text

for i in range(args.steps):
    onehot = py_train_dataset._get_onehot(currc).unsqueeze(0)
    logprob, hidden_state = model(onehot, hidden_state)

    currc = py_train_dataset.i2c[logprob.argmax(dim=2).item()]
    generated.append(currc)

with open(name, "w") as fp:
    fp.write("".join(generated))
