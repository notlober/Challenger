import os
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)

MODEL_NAME = "Qwen/Qwen2.5-7B"
DATASET_NAME = "HuggingFaceFW/fineweb-2"
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.01
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

MAX_LENGTH = 512
MAX_STEPS = 1000
WARMUP_STEPS = 50

LOGGING_STEPS = 1
SAVE_STEPS = MAX_STEPS
SAVE_TOTAL_LIMIT = 1

OUTPUT_DIR = "./simple_llm_trainer_output"
SEED = 42
set_seed(SEED)

class SimpleStreamingDataset(IterableDataset):
    def __init__(self, iterable_dataset, tokenizer, max_length):
        self.iterable_dataset = iterable_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos = not getattr(tokenizer, 'add_eos_token', False)
        if self.tokenizer.pad_token is None:
             if self.tokenizer.eos_token is not None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
             else:
                 logger.warning("Tokenizer does not have a pad_token and eos_token is None. Padding might not work correctly.")


    def __iter__(self):
        token_buffer = []
        dataset_iterator = iter(self.iterable_dataset)

        while True:
            while len(token_buffer) < self.max_length:
                try:
                    item = next(dataset_iterator)
                    text = item["text"]
                    if text:
                        encoded = self.tokenizer.encode(text, add_special_tokens=False)
                        if self.add_eos and self.tokenizer.eos_token_id is not None:
                             encoded.append(self.tokenizer.eos_token_id)

                        token_buffer.extend(encoded)
                except StopIteration:
                    logger.info("Dataset stream finished.")
                    return

            token_chunk = token_buffer[:self.max_length]
            yield {"input_ids": torch.tensor(token_chunk, dtype=torch.long),
                   "labels": torch.tensor(token_chunk, dtype=torch.long)}

            token_buffer = token_buffer[self.max_length:]

def load_components(model_name, dataset_name, max_length):
    logger.info(f"Loading streaming dataset {dataset_name}...")
    train_dataset_iterable = load_dataset(dataset_name, 'tur_Latn', split="train", streaming=True, trust_remote_code=True)
    logger.info(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set tokenizer pad_token to eos_token ({tokenizer.eos_token_id})")


    logger.info("Initializing SimpleStreamingDataset (Iterable)...")
    train_dataset = SimpleStreamingDataset(train_dataset_iterable, tokenizer, max_length=max_length)

    logger.info(f"Initializing model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    params = sum(p.numel() for p in model.parameters())/1_000_000
    logger.info(f"Model parameters: {params:.2f}M")

    return model, train_dataset, tokenizer

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("--- Simple Hardcoded LLM Training Demo with Hugging Face Trainer ---")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info(f"Learning Rate: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
    logger.info(f"Per Device Batch Size: {PER_DEVICE_BATCH_SIZE}")
    logger.info(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Effective Batch Size (per GPU): {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Sequence Length: {MAX_LENGTH}")
    logger.info(f"Max Steps: {MAX_STEPS}")
    logger.info(f"Warmup Steps: {WARMUP_STEPS}")
    logger.info(f"Logging Steps: {LOGGING_STEPS}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Seed: {SEED}")
    logger.info("--------------------------------------------------------------------")


    model, train_dataset, tokenizer = load_components(
        MODEL_NAME, DATASET_NAME, MAX_LENGTH
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

    logger.info("Training finished.")
    logger.info(f"Final model and tokenizer saved to {os.path.join(OUTPUT_DIR, 'final_model')}")