from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/gemini/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_file = request.files['image']
        img = Image.open(io.BytesIO(image_file.read()))

        # Directly inline the image wrapping
        response = model.generate_content([
            "Describe this image in detail",
            genai.types.Part.from_image(img)
        ])

        return jsonify({
            "description": response.text,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
