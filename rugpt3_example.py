from transformers import GPT2LMHeadModel, GPT2Tokenizer, \
    TextDataset, DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments
import torch
DEVICE = torch.device("cuda:0")

# Load GPT-3 on Sber
model_name_or_path = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path,
                                          cache_dir="rugpt3small_based_on_gpt2_cached_token/")
model = GPT2LMHeadModel.from_pretrained(model_name_or_path,
                                        cache_dir="rugpt3small_based_on_gpt2_cached_model/").to(DEVICE)

# Load train data text
train_path = 'pchelovod_junior.txt'

# Create dataset
train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=64)

# Create dataloader (slice text on part)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./finetuned",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=50,  # number of training epochs
    per_device_train_batch_size=16,  # batch size for training. Optimal 32. If 'RuntimeError: CUDA out of memory. Tried to allocate...' set size to 16
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=10,  # number of warmup steps for learning rate scheduler
    gradient_accumulation_steps=16,  # to make "virtual" batch size larger
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    optimizers=(torch.optim.AdamW(model.parameters(), lr=1e-5), None)  # Optimizer and lr scheduler
)

# Run train
trainer.train()

# Save trained model
torch.save(model, 'my_model')

# Load model
# torch.load('my_model')

text = "Example text  "
input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
model.eval()
with torch.no_grad():
    out = model.generate(input_ids,
                        do_sample=True,
                        num_beams=2,
                        temperature=1.7,
                        top_p=0.7,
                        max_length=100,
                        )

generated_text = list(map(tokenizer.decode, out))[0]

# Generated text
print(f'\n{generated_text}')

