# Kafka GPT-2 Story Generator

A fine-tuned GPT-2 model trained to generate Kafka-style stories, capturing the essence of Franz Kafka's unique writing style, existential themes, and surreal narratives.

## Model Availability

This model is now available on Hugging Face at: [bankar404/kafka-gpt2-story-generator](https://huggingface.co/bankar404/kafka-gpt2-story-generator)

You can download and use the model directly from Hugging Face using the transformers library.

## Model Description

This model is a fine-tuned version of GPT-2, specifically trained on Franz Kafka's complete works to generate stories that mimic his distinctive writing style. The model captures key elements of Kafka's writing, including:
- Existential themes
- Bureaucratic absurdity
- Psychological depth
- Surreal transformations
- Complex sentence structures

### Model Type
- **Architecture:** GPT-2 (fine-tuned)
- **Language:** English
- **License:** MIT
- **Fine-tuned from:** GPT-2
- **Repository:** [bankar404/kafka-gpt2-story-generator](https://huggingface.co/bankar404/kafka-gpt2-story-generator)

## Intended Use

This model is designed for:
- Generating Kafka-style short stories
- Creative writing assistance
- Literary experimentation
- Educational purposes
- Research in style transfer

### Primary Use Cases
1. Story Generation
   - Generate complete short stories
   - Create story continuations
   - Develop Kafka-style narratives

2. Creative Writing
   - Assist writers in developing Kafka-esque themes
   - Provide inspiration for surreal narratives
   - Help understand Kafka's writing style

3. Educational
   - Study Kafka's writing techniques
   - Analyze narrative structures
   - Explore existential themes

## Training Data

The model was fine-tuned on:
- Complete works of Franz Kafka
- Including novels, short stories, and letters
- Focus on his most characteristic works

## Performance

The model excels at:
- Maintaining Kafka's distinctive writing style
- Generating coherent narratives
- Creating existential themes
- Producing complex sentence structures

### Limitations
- May occasionally generate repetitive patterns
- Requires careful prompt engineering
- Best results with English prompts
- May not capture all nuances of Kafka's style

## Usage

### Python Code Example
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("bankar404/kafka-gpt2-story-generator")
tokenizer = GPT2Tokenizer.from_pretrained("bankar404/kafka-gpt2-story-generator")

# Generate text
prompt = "A man wakes up one morning to find himself transformed"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    max_length=500,
    temperature=0.85,
    top_k=50,
    top_p=0.92,
    num_return_sequences=1
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Recommended Generation Parameters
- Temperature: 0.85
- Top-k: 50
- Top-p: 0.92
- Max length: 500-1000 tokens
- Repetition penalty: 1.1

## Ethical Considerations

This model should be used responsibly:
- Not for generating harmful or offensive content
- Respecting copyright and intellectual property
- Being transparent about AI-generated content
- Using for educational and creative purposes

## Citation

If you use this model in your research or work, please cite:
```bibtex
@misc{kafka-gpt2-story-generator,
  author = {Bankar404},
  title = {Kafka GPT-2 Story Generator},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/bankar404/kafka-gpt2-story-generator}}
}
```

## License

This model is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Franz Kafka for his literary works
- OpenAI for the GPT-2 model
- Hugging Face for the transformers library and model hosting
- The open-source community for their contributions

## Contact

For questions or feedback about this model, please open an issue on the [Hugging Face repository](https://huggingface.co/bankar404/kafka-gpt2-story-generator). 