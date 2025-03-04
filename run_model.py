import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

logger = logging.getLogger(__name__)

def generate_text(prompt, tokenizer, model, max_length=800, temperature=0.85, top_k=50, top_p=0.92):
    """
    Generate text with optimized parameters for faster generation (30-45s)
    """
    try:
        # Optimized tokenization settings
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1500,  # Reduced for faster processing
            return_attention_mask=True
        )
        
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        # Performance optimized generation parameters
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_length=800,      # Reduced for faster generation
            min_length=100,      # Reduced minimum
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            num_beams=2,         # Reduced beam size for speed
            do_sample=True,
            num_return_sequences=1,
            early_stopping=True,
            length_penalty=1.0,
            use_cache=True
        )

        # Efficient decoding
        if len(output) > 0 and output[0] is not None:
            return tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            return "Error: Failed to generate text."

    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        return f"Error generating text: {str(e)}"

def load_model_and_tokenizer():
    # Load Model & Tokenizer
    model_path = "./model"  # Updated path to the new model directory
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Performance optimizations
    model.eval()
    torch.set_grad_enabled(False)  # Disable gradient computation
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Optional: Half precision for faster GPU inference
        if torch.cuda.get_device_properties(0).total_memory >= 6000000000:  # 6GB+ GPU
            model = model.half()  # Convert to FP16
    else:
        model = model.to("cpu")
        # CPU optimizations
        torch.set_num_threads(4)  # Adjust based on CPU cores
        torch.set_num_interop_threads(1)  # Reduce inter-op parallelism

    return tokenizer, model

if __name__ == "__main__":
    # Interactive Prompt
    tokenizer, model = load_model_and_tokenizer()
    while True:
        prompt = input("Enter a prompt: ")
        if not prompt.strip():
            break
        print("\n📜 **Generated Short Story:**\n", generate_text(prompt, tokenizer, model))
