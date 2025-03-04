# Kafka GPT-2 Project

This project implements a GPT-2 based text generation system with a web interface. It uses a fine-tuned GPT-2 model trained on Kafka's works to generate Kafka-style text.

## Features
- Web-based interface for text generation
- Fine-tuned GPT-2 model on Kafka's writing style
- Real-time text generation
- Customizable generation parameters
- Text-to-speech functionality for generated stories

## Project Structure
```
project/
├── app.py                  # Main Flask application
├── run_model.py           # Model inference code
├── storybot.py           # Story generation utilities
├── model/                # All model-related files
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.json
│   ├── generation_config.json
│   ├── merges.txt
│   └── README.md         # Model documentation
├── static/               # Static assets
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   ├── images/          # Image assets
│   └── audio/           # Generated audio files (created on-demand)
├── templates/            # HTML templates
├── components/          # Reusable components
├── requirements.txt     # Python dependencies
└── .env                # Environment variables
```

## Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the model files:
   - The model is now available on Hugging Face at [bankar404/kafka-gpt2-story-generator](https://huggingface.co/bankar404/kafka-gpt2-story-generator)
   - You can download the model directly using the transformers library
   - For detailed model information, see [model/README.md](model/README.md)

3. Run the application:
```bash
python app.py
```

## Environment Variables
Create a `.env` file with the following variables:
- Add your required environment variables here

## Model Information
The project uses a fine-tuned GPT-2 model trained on Kafka's works. For detailed information about the model, including:
- Model architecture and capabilities
- Training data and methodology
- Usage examples and parameters
- Performance characteristics
- Ethical considerations

Please refer to the [model documentation](model/README.md) or visit the [Hugging Face repository](https://huggingface.co/bankar404/kafka-gpt2-story-generator).

## Audio Files
- Audio files are generated on-demand when users request text-to-speech
- Files are stored in `static/audio/` directory
- Supported formats: MP3, WAV, OGG, M4A
- Audio files are not included in version control

## Note
The model is now hosted on Hugging Face and can be downloaded directly using the transformers library. No manual download of model files is required.