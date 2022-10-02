[<img src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg" alt="Python 3.7 3.8 3.9">](https://pypi.org/project/pytorch-widedeep/)
[<img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">](https://github.com/psf/black)
[<img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance">](https://github.com/jrzaurin/pytorch-widedeep/graphs/commit-activity)
[<img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat" alt="contributions welcome">](https://github.com/jrzaurin/pytorch-widedeep/issues)
## LexGen

Documentation: https://masslightsquared.github.io/lex_gen/

### Virtual environment

```bash
python -m venv venv
& venv/scripts/activate
python -m pip install pip setuptools wheel
python -m pip install -e .
```

### Install PyTorch CUDA

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### To-Do:

#### Training Loop

1. Setup pipeline for tracking model metrics and hyperparameters in our designated log (TrainingArguments)
2. ✔ Write function train() which encapsulates the training loop
3. Set-up W&B integrations
4. Set-up Optuna integrations
5. Obtain optimal training parameters
6. Set-up training of "[gpt2-large](https://huggingface.co/transformers/v2.2.0/pretrained_models.html)" on [Lambda cloud GPUs](https://lambdalabs.com/)
7. Write customized training loop (migrate from Hugging Face Training class)

#### Model

1. Implement a learning rate schedular
2. Implement customized weight clipping
3. Implement customized optimizer
4. Implement key-token weighted loss function
5. Optimize hyper-parameters
6. ✔ Train model for 10 epochs to reduce loss below 3.0

#### Inference

1. Truncate printed text to last punctuation mark (to prevent user from seeing incomplete sentences)
2. Extend the length of the returned prompt

#### Serving

1. Create CLI using [Typer](https://typer.tiangolo.com/)
2. Implement Real-time serving using [FastAPI](https://fastapi.tiangolo.com/#typer-the-fastapi-of-clis)

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```

3. Complete integration with model server (perhaps W&B or MLFLow?)

#### Packaging

1. Package the application using Docker container engine.

...

