from operations.utils import read_text
from operations.data import tokenize_text, get_tokenizer
from operations.data import data_split
from operations.data import stage_data
from config.config import logger
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, ProgressCallback
import transformers
import torch

args = TrainingArguments(
    output_dir="artifacts",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=3000,
    logging_strategy="epoch",
    logging_dir='logs/',
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_strategy="steps",
    save_steps=200,
    fp16=True,
    push_to_hub=False,
)

# block size
args.context_length = 256

class PrinterCallback(ProgressCallback):
    "A callback that logs a message at the end of each training epoch"
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)
            # logger.info(args)
            logger.info(state)
            print(logs)


def train(args: transformers.TrainingArguments, trial: bool=False, optimize=False):
    """Training loop for the fine-tuning of model parameters.

    Args:
        args (transformers.TrainingArguments, optional): _description_. Defaults to args.
        trial (bool, optional): _description_. Defaults to False.
        optimize (bool, optional): _description_. Defaults to False.
    """

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = tokenize_text(read_text())
    xtrain, train_mask, xdev, dev_mask = data_split(inputs)

    if optimize == True:
        train_loader, dev_loader = stage_data(xtrain[:100], train_mask[:100], xdev[:100], dev_mask[:100])
    else:
        train_loader, dev_loader = stage_data(xtrain, train_mask, xdev, dev_mask)

    # Tokenization
    tokenizer = get_tokenizer()
    config = AutoConfig.from_pretrained(
        tokenizer.name_or_path, 
        vocab_size=len(tokenizer),
        n_ctx=args.context_length, 
        bos_token_id=tokenizer.bos_token_id, 
        eos_token_id=tokenizer.eos_token_id
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Model
    model = GPT2LMHeadModel.from_pretrained(tokenizer.name_or_path, config=config)
    model_size = sum(t.numel() for t in model.parameters())
    logger.info(f"number of model parameters: {model_size/1000**2:.1f}M")
    
    # Training Loop
    callback = PrinterCallback()
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_loader,
        eval_dataset=dev_loader,
        callbacks=[callback]
    )
    
    # Train
    trainer.train()