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
│   └── merges.txt
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
   - Due to size limitations, model files are not included in this repository
   - You can download the required files from [link to your chosen hosting service]
   - Place all downloaded files in the `model/` directory

3. Run the application:
```bash
python app.py
```

## Environment Variables
Create a `.env` file with the following variables:
- Add your required environment variables here

## Model Files
This project uses a fine-tuned GPT-2 model. The model files are not included in this repository due to size limitations. You have two options to get the model:

1. Download pre-trained model files:
   - [Add link to where you'll host the model files]
   - Place the downloaded files in the `model/` directory

2. Train your own model:
   - [Add instructions if you want to include how to train the model]

## Audio Files
- Audio files are generated on-demand when users request text-to-speech
- Files are stored in `static/audio/` directory
- Supported formats: MP3, WAV, OGG, M4A
- Audio files are not included in version control

## Note
Make sure to download all required model files before running the application. The total size of model files is approximately 477MB.