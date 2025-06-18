import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import json
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone()
        }

def test_peft_training():
    print("🧪 Testing PEFT Training Setup...")
    
    # Load model and tokenizer
    model_name = "microsoft/DialoGPT-medium"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for stability
        trust_remote_code=True
    )
    
    print("✅ Model loaded successfully")
    
    # Enable gradients for input embeddings
    model.enable_input_require_grads()
    print("✅ Input gradients enabled")
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        fan_in_fan_out=False,
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.enable_input_require_grads()
    
    print("✅ LoRA setup completed")
    
    # Create minimal dataset
    sample_data = [
        "User: Hello, how are you? Assistant: I'm doing well, thank you for asking!",
        "User: What's the weather like? Assistant: I don't have real-time weather data, but I can help you find weather information.",
        "User: Can you help me? Assistant: Of course! I'm here to help. What do you need assistance with?"
    ]
    
    dataset = SimpleDataset(sample_data, tokenizer)
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        logging_steps=1,
        save_steps=100,
        eval_strategy="no",
        save_total_limit=1,
        fp16=False,  # Disable fp16 for stability
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=None,
        max_steps=3,  # Just 3 steps for testing
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("✅ Trainer created")
    
    # Test training
    try:
        print("🚀 Starting test training...")
        trainer.train()
        print("🎉 Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = test_peft_training()
    if success:
        print("\n✅ PEFT training is working! We can proceed with full training.")
    else:
        print("\n❌ PEFT training failed. Need to investigate further.") 