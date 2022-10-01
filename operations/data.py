from transformers import GPT2Tokenizer, GPT2Config, GPT2ForSequenceClassification, GPT2TokenizerFast, BatchEncoding
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from config.config import logger
import torch
import typing

def tokenize_text(text: typing.Union[str, typing.List[str], typing.List[typing.List[str]]] = None
    , model_name: str="gpt2", padding_side: str="right", context_length: int=256, tokenizer_only: bool=False) -> BatchEncoding:
    """Prepares inputs for a model. Utilizes Hugging Face's PreTrainedTokenizer base class.

    Args:
        text (str): The sequence or batch of sequences to be encoded.
        model_name (str, optional): Model name. Defaults to "gpt2".
        padding_side (str, optional): The default value for the side on which the model should have padding applied. Should be 'right' or 'left'. Defaults to "right".
        context_length (int, optional): The length of input sequences fed to the model. Defaults to 256.
        tokenizer_only (bool, optional): Returns only the instantiated tokenizer class when set to 'True'. Defaults to False.

    Returns:
        BatchEncoding: (input_ids, attention_mask...)
    """
    
    # Instantiate tokenizer and pass `gpt2` to the `from_pretrained` method 
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    # Select token to uses as `pad_token`
    tokenizer.pad_token = tokenizer.eos_token
    # Default to right padding
    tokenizer.padding_side = padding_side
    # Set context length
    context_length = context_length
    
    if tokenizer_only == True:
        return tokenizer

    logger.info(tokenizer)
    
    # Process text
    inputs = tokenizer(text, 
                       padding='longest',
                       truncation=True,
                       return_tensors="pt",
                       max_length=context_length, # context size GPT-2: 1,024, GPT-3: 2,048
                       return_overflowing_tokens=True, # tokenize input and split into chunks
                       return_length=True, # return length of each created chunk
                       return_special_tokens_mask=True
                      )
    
    logger.info(f"Input IDs length: {len(inputs['input_ids'])}")
    logger.info(f"Input chunk lengths: {(inputs['length'])}")
    logger.info(f"Chunk mapping: {inputs['overflow_to_sample_mapping']}")

    return inputs


class DataLoads(Dataset):
    """A barebones dataloader class for PyTorch."""
    def __init__(self, X, Mask):
        self.x = X
        self.mask = Mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'input_ids':self.x[idx],
            'attention_mask':self.mask[idx],
        }


def data_split(tokens: Dict , train_split: float=0.8, validation_split: float=0.9) -> torch.Tensor:
    """Split inputs into train and dev sets.

    Args:
        tokens (Dict): _description_
        train_split (float, optional): _description_. Defaults to 0.8.
        validation_split (float, optional): _description_. Defaults to 0.9.

    Returns:
        torch.Tensor: _description_
    """
    assert "input_ids" and "attention_mask" in tokens.keys()

    n1 = int(train_split * len(tokens['input_ids']))
    n2 = int(validation_split * len(tokens['input_ids']))

    Xtr = tokens['input_ids'][:n1]
    tr_mask = tokens['attention_mask'][:n1]

    Xval = tokens['input_ids'][n1:n2]
    val_mask = tokens['attention_mask'][n1:n2]

    logger.info(f"{Xtr.shape=}")
    logger.info(f"{Xval.shape=}")

    return Xtr, tr_mask, Xval, val_mask


def stage_data(xtrain: torch.tensor,
               xtrain_mask: torch.tensor, 
               xval: torch.tensor, 
               xval_mask: torch.tensor) -> Dataset:
    """Loads tokenized data to customized DataLoads class for input to model.

    Args:
        xtrain (torch.tensor): _description_
        xtrain_mask (torch.tensor): _description_
        xval (torch.tensor): _description_
        xval_mask (torch.tensor): _description_

    Returns:
        Dataset: _description_
    """
    train_loader = DataLoads(xtrain, xtrain_mask)
    val_loader = DataLoads(xval, xval_mask)
    
    return train_loader, val_loader


def get_tokenizer(model_name: str="gpt2", padding_side:str="right") -> GPT2TokenizerFast:
    """Instantiates tokenizer.

    Args:
        model_name (str, optional): _description_. Defaults to "gpt2".
        padding_side (str, optional): _description_. Defaults to "right".

    Returns:
        GPT2TokenizerFast: Instantiated tokenizer class.
    """
    # Instantiate tokenizer and pass `gpt2` to the `from_pretrained` method 
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    # Select token to uses as `pad_token`
    tokenizer.pad_token = tokenizer.eos_token
    # Default to right padding
    tokenizer.padding_side = padding_side
    
    return tokenizer  