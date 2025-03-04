import logging
import requests
from flask import Flask, render_template, request, send_file, Response, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from run_model import generate_text, load_model_and_tokenizer
import os
from dotenv import load_dotenv
from gtts import gTTS
from pathlib import Path
import json
import threading
import queue
from collections import defaultdict, Counter
import re
import markdown
import pdfkit
from io import BytesIO

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load Model & Tokenizer
tokenizer, model = load_model_and_tokenizer()
logger.info("Model and tokenizer loaded successfully.")

# Load environment variables
load_dotenv()

# Create a directory for audio files if it doesn't exist
AUDIO_DIR = Path("static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Simple in-memory analytics storage
analytics = {
    'story_count': 0,
    'genre_counts': defaultdict(int),
    'avg_length': 0,
    'total_words': 0,
    'reading_tracker': [],
    'literary_metrics': [],
    'kafka_score_avg': 0
}
analytics_lock = threading.Lock()

def analyze_literary_quality(text):
    """Analyze the literary quality of generated text"""
    # Kafka-specific keywords and themes
    kafka_themes = {
        'bureaucratic': ['office', 'bureaucracy', 'official', 'authority', 'document'],
        'existential': ['meaningless', 'absurd', 'existence', 'alienation', 'isolation'],
        'psychological': ['anxiety', 'fear', 'guilt', 'dream', 'nightmare'],
        'metamorphosis': ['transform', 'change', 'creature', 'insect', 'body'],
    }
    
    # Initialize metrics
    metrics = {
        'kafka_score': 0,
        'theme_presence': {},
        'complexity': 0,
        'style_match': 0
    }
    
    # Calculate theme presence
    text_lower = text.lower()
    for theme, keywords in kafka_themes.items():
        theme_score = sum(text_lower.count(keyword) for keyword in keywords)
        metrics['theme_presence'][theme] = min(theme_score * 20, 100)
    
    # Calculate Kafka score
    metrics['kafka_score'] = sum(metrics['theme_presence'].values()) / len(kafka_themes)
    
    # Calculate complexity
    sentences = text.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    metrics['complexity'] = min(avg_sentence_length * 5, 100)
    
    # Calculate style match
    long_sentences = sum(1 for s in sentences if len(s.split()) > 20)
    metrics['style_match'] = min(long_sentences * 10, 100)
    
    return metrics

def update_analytics(story_text, genre):
    """Update analytics data with literary metrics"""
    global analytics
    with analytics_lock:
        analytics['story_count'] += 1
        analytics['genre_counts'][genre] += 1
        words = len(story_text.split())
        analytics['total_words'] += words
        analytics['avg_length'] = analytics['total_words'] / analytics['story_count']
        
        # Add literary quality metrics
        metrics = analyze_literary_quality(story_text)
        analytics['literary_metrics'].append(metrics)
        
        # Update average Kafka score
        total_kafka_score = sum(m['kafka_score'] for m in analytics['literary_metrics'])
        analytics['kafka_score_avg'] = total_kafka_score / len(analytics['literary_metrics'])
        
        logger.info(f"Updated analytics with literary metrics: {metrics}")

def enhance_with_moonshot(text):
    """
    Enhance the generated text using Moonshot API to make it more Kafka-like
    and grammatically correct with proper story structure.
    """
    MOONSHOT_API_KEY = os.getenv('MOONSHOT_API_KEY')
    MOONSHOT_API_URL = "https://api.moonshot.ai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {MOONSHOT_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    As a Kafka-style story enhancer, complete and refine the following story fragment. 
    Add appropriate beginning and ending, maintain Kafka's distinctive style, and ensure 
    grammatical correctness while preserving the core narrative:

    Original story fragment:
    {text}

    Please provide a complete, well-structured story that:
    1. Has a proper beginning that sets the scene
    2. Maintains Kafka's surreal and existential themes
    3. Includes proper paragraph breaks
    4. Has a meaningful conclusion
    5. Is grammatically correct
    """

    data = {
        "model": "moonshot-v1-8k",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in Kafka's writing style and story structure."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    try:
        response = requests.post(MOONSHOT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        logger.debug(f"Moonshot API Response Status: {response.status_code}")
        logger.debug(f"Moonshot API Response: {response.text[:200]}...")
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            enhanced_story = result['choices'][0]['message']['content']
            return enhanced_story
        else:
            logger.error(f"Unexpected API response structure: {result}")
            return text
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Moonshot API request error: {str(e)}")
        return text
    except Exception as e:
        logger.error(f"Moonshot API error: {str(e)}")
        return text

def post_process_text(text):
    """
    Post-process the generated text using Moonshot API for Kafka-style enhancement
    """
    return enhance_with_moonshot(text)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        genre = request.form.get('genre', 'general')
        length = request.form.get('length', 'medium')
        mood = request.form.get('darkness', '50')
        
        # Modify the prompt based on settings
        enhanced_prompt = f"""Generate a {length} {genre} story with a mood level of {mood}/100.
        Story should be {
            'around 500 words' if length == 'short' 
            else 'around 1000 words' if length == 'medium'
            else 'around 2000 words'
        }.
        Theme: {prompt}"""
        
        try:
            # Add length validation
            if not enhanced_prompt or len(enhanced_prompt.strip()) == 0:
                return render_template('index.html', 
                                    generated_text="Please enter a valid prompt")
            
            # Calculate token length and truncate if necessary
            enhanced_prompt = enhanced_prompt.strip()[:500]  # Limit prompt length
            
            # Generate initial text
            generated_text = generate_text(enhanced_prompt, tokenizer, model)
            
            # Check if generation was successful
            if generated_text.startswith("Error"):
                return render_template('index.html', 
                                    generated_text="Failed to generate story. Please try again with a shorter prompt.")
            
            # Post-process the text
            enhanced_text = post_process_text(generated_text)
            
            # Add formatting for better readability
            formatted_text = enhanced_text.replace("\n", "<br>")
            
            # Update analytics after successful generation
            update_analytics(enhanced_text, genre)
            
            return render_template('index.html', 
                                 generated_text=formatted_text, 
                                 raw_text=generated_text)
        except Exception as e:
            logger.error(f"Error in index route: {e}")
            return render_template('index.html', 
                                 generated_text="An error occurred while processing your request.")
    return render_template('index.html')

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    """Generate audio file from text using Google TTS"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        lang = data.get('lang', 'en')
        slow = data.get('slow', False)

        if not text:
            return {"error": "No text provided"}, 400

        # Create a unique filename
        audio_file = AUDIO_DIR / f"story_{hash(text)}.mp3"
        
        if not audio_file.exists():
            # Add pauses for better narration
            processed_text = text.replace('. ', '... ')
            processed_text = processed_text.replace('! ', '!... ')
            processed_text = processed_text.replace('? ', '?... ')
            
            # Generate audio with Google TTS
            tts = gTTS(text=processed_text, lang=lang, slow=slow)
            tts.save(str(audio_file))

        return {"audio_url": f"/static/audio/{audio_file.name}"}
        
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return {"error": str(e)}, 500

@app.route('/analytics')
def view_analytics():
    """View enhanced analytics dashboard"""
    # Calculate aggregate metrics
    if analytics['literary_metrics']:
        latest_metrics = analytics['literary_metrics'][-1]
        avg_metrics = {
            'kafka_score': analytics['kafka_score_avg'],
            'theme_presence': {},
            'complexity': sum(m['complexity'] for m in analytics['literary_metrics']) / len(analytics['literary_metrics']),
            'style_match': sum(m['style_match'] for m in analytics['literary_metrics']) / len(analytics['literary_metrics'])
        }
        
        # Calculate average theme presence
        themes = set()
        for metrics in analytics['literary_metrics']:
            themes.update(metrics['theme_presence'].keys())
        
        for theme in themes:
            scores = [m['theme_presence'].get(theme, 0) for m in analytics['literary_metrics']]
            avg_metrics['theme_presence'][theme] = sum(scores) / len(scores)
    else:
        latest_metrics = None
        avg_metrics = None
    
    return render_template('analytics.html', 
                         analytics=analytics,
                         latest_metrics=latest_metrics,
                         avg_metrics=avg_metrics)

@app.route('/add-reading', methods=['POST'])
def add_reading():
    """Add a book to the reading tracker"""
    book_title = request.form.get('book_title')
    author = request.form.get('author')
    pages = request.form.get('pages', type=int)
    time_spent = request.form.get('time_spent', type=int)  # in minutes

    if book_title and author:
        analytics['reading_tracker'].append({
            'title': book_title,
            'author': author,
            'pages': pages,
            'time_spent': time_spent
        })
        return jsonify(success=True, message="Book added to reading tracker.")
    return jsonify(success=False, message="Failed to add book.")

@app.route('/export-story', methods=['POST'])
def export_story():
    """Export story in various formats"""
    try:
        data = request.get_json()
        story = data.get('story', '')
        format = data.get('format', 'txt')
        filename = data.get('filename', 'kafka_story')

        if not story:
            return "No story content provided", 400

        if format == 'md':
            # For markdown, just add some basic formatting
            md_content = f"""# {filename}\n\n{story}"""
            return Response(
                md_content,
                mimetype='text/markdown',
                headers={'Content-Disposition': f'attachment;filename={filename}.md'}
            )
        
        elif format == 'pdf':
            # Convert to HTML first
            html_content = f"""
            <html>
                <head>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            line-height: 1.6;
                            margin: 2cm;
                        }}
                        h1 {{
                            color: #2c3e50;
                            text-align: center;
                        }}
                    </style>
                </head>
                <body>
                    <h1>{filename}</h1>
                    {story.replace('\n', '<br>')}
                </body>
            </html>
            """
            
            # List of common wkhtmltopdf installation paths
            possible_paths = [
                r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe',
                r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe',
                r'C:\wkhtmltopdf\bin\wkhtmltopdf.exe',
                r'wkhtmltopdf'  # If it's in system PATH
            ]

            pdf = None
            last_error = None

            # Try each possible path
            for path in possible_paths:
                try:
                    if path == 'wkhtmltopdf':
                        pdf = pdfkit.from_string(html_content, False)
                    else:
                        config = pdfkit.configuration(wkhtmltopdf=path)
                        pdf = pdfkit.from_string(html_content, False, configuration=config)
                    break
                except Exception as e:
                    last_error = e
                    continue

            if pdf is None:
                logger.error(f"Failed to generate PDF. Error: {last_error}")
                return "Could not generate PDF. Please make sure wkhtmltopdf is installed.", 500

            return send_file(
                BytesIO(pdf),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'{filename}.pdf'
            )
        
        else:
            return "Unsupported format", 400

    except Exception as e:
        logger.error(f"Error in export_story: {e}")
        return str(e), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, use_reloader=False)
    logger.info("Flask server stopped.") 