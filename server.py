import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Prevent online lookups

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
import io
import pickle
import logging
from typing import Tuple
from transformers import BlipProcessor, BlipForConditionalGeneration
from safetensors.torch import load_file
import torch

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)

# Model configuration
MODELS = {
    'from_scratch': {
        'model_path': 'models/from_scratch_model.keras',
        'feature_extractor_path': 'models/feature_extractor.keras',
        'tokenizer_path': 'models/tokenizer.pkl',
        'max_length': 34,
        'img_size': 224,
        'model_type': 'keras'
    },
    'blip-finetuned': {
        'model_weights': 'models/model.safetensors',
        'processor_dir': 'models/blip_processor',
        'model_type': 'blip'
    }
}

def load_blip_components(config):
    """Load BLIP model with correct weight paths"""
    try:
        processor_dir = config['processor_dir']
        weights_path = config['model_weights']
        
        # Verify required files exist
        required_processor_files = ['preprocessor_config.json', 
                                  'special_tokens_map.json',
                                  'tokenizer_config.json']
        
        for f in required_processor_files:
            if not os.path.exists(os.path.join(processor_dir, f)):
                raise FileNotFoundError(f"Missing processor file: {f}")
        
        # Verify weights file exists
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        # Load processor
        processor = BlipProcessor.from_pretrained(
            processor_dir,
            use_fast=False,
            local_files_only=True
        )
        
        # FIRST load the base model architecture
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",  # Load original config
            local_files_only=False  # Need to download config first time
        )
        
        # THEN load your fine-tuned weights
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict , strict=False)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return processor, model
        
    except Exception as e:
        logging.error(f"BLIP loading failed: {str(e)}")
        raise
def load_model_components():
    loaded = {}
    for name, config in MODELS.items():
        try:
            if config['model_type'] == 'keras':
                loaded[name] = {
                    'model': load_model(config['model_path']),
                    'feature_extractor': load_model(config['feature_extractor_path']),
                    'tokenizer': pickle.load(open(config['tokenizer_path'], 'rb')),
                    'max_length': config['max_length'],
                    'img_size': config['img_size'],
                    'model_type': 'keras'
                }
            elif config['model_type'] == 'blip':
                processor, model = load_blip_components(config)
                loaded[name] = {
                    'processor': processor,
                    'model': model,
                    'model_type': 'blip'
                }
            logging.info(f"Successfully loaded {name} model")
        except Exception as e:
            logging.error(f"Failed loading {name} model: {str(e)}")
            raise
    return loaded

# Rest of your code remains the same...

MODEL_CACHE = load_model_components()

def generate_caption(image_bytes: bytes, model_type: str) -> Tuple[str, str]:
    try:
        model_data = MODEL_CACHE[model_type]
        
        if model_data['model_type'] == 'keras':
            # Original Keras model processing
            img = load_img(io.BytesIO(image_bytes), target_size=(model_data['img_size'], model_data['img_size']))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            image_features = model_data['feature_extractor'].predict(img, verbose=0)

            in_text = "startseq"
            for _ in range(model_data['max_length']):
                sequence = model_data['tokenizer'].texts_to_sequences([in_text])[0]
                sequence = pad_sequences([sequence], maxlen=model_data['max_length'])
                yhat = model_data['model'].predict([image_features, sequence], verbose=0)
                yhat_index = np.argmax(yhat)
                word = model_data['tokenizer'].index_word.get(yhat_index, None)
                if word is None:
                    break
                in_text += " " + word
                if word == "endseq":
                    break
            caption = in_text.replace("startseq", "").replace("endseq", "").strip()
            
        elif model_data['model_type'] == 'blip':
            
            raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Process image and generate caption
            inputs = model_data['processor'](raw_image, return_tensors="pt")
            with torch.no_grad():
                outputs = model_data['model'].generate(**inputs)
            
            caption = model_data['processor'].decode(outputs[0], skip_special_tokens=True)
            
        return caption, model_type
        
    except Exception as e:
        logging.error(f"Generation error: {str(e)}", exc_info=True)
        raise RuntimeError(f"Caption generation failed: {str(e)}")

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "loaded_models": list(MODEL_CACHE.keys())
    })

@app.route('/api/caption', methods=['POST'])
def handle_caption_request():
    if 'file' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Normalize any variation to 'blip-finetuned'
        requested_model = request.form.get('model', 'from_scratch').lower()
        model_type = 'blip-finetuned' if any(x in requested_model for x in ['blip', 'b']) else 'from_scratch'
        
        if model_type not in MODEL_CACHE:
            return jsonify({
                "status": "error",
                "error": f"Model '{requested_model}' not available",
                "available_models": list(MODEL_CACHE.keys()),
                "solution": f"Use either: {list(MODEL_CACHE.keys())}"
            }), 400
            
        image_file = request.files['file']
        if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({"error": "Only JPG/PNG images supported"}), 400
        
        caption, model_used = generate_caption(image_file.read(), model_type)
        
        return jsonify({
            "status": "success",
            "model_used": model_used,
            "caption": caption,
            "image_size": MODEL_CACHE[model_used]['img_size'] if model_used == 'from_scratch' else "variable"
        })
        
    except Exception as e:
        logging.error(f"API Error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "supported_models": list(MODEL_CACHE.keys())
        }), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    MODEL_CACHE = load_model_components()
