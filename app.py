from flask import Flask, request, jsonify, render_template
import boto3
import requests
import logging
import traceback
import time
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# AWS Configuration
rekognition = boto3.client('rekognition', region_name='us-east-1')
s3 = boto3.client('s3', region_name='us-east-1')
BUCKET_NAME = 'projectappaws'  # Your actual S3 bucket name

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
OPENAI_API_KEY = OPENAI_API_KEY.strip()
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'

# Rate limiting variables
last_request_time = 0
REQUEST_INTERVAL = 1  # Minimum interval between requests in seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    global last_request_time
    try:
        current_time = time.time()
        if current_time - last_request_time < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - (current_time - last_request_time))
        last_request_time = time.time()

        file = request.files['image']
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        file_name = f"images/{file.filename}"
        s3.upload_fileobj(file, BUCKET_NAME, file_name)
        image_url = s3.generate_presigned_url('get_object',
                                              Params={'Bucket': BUCKET_NAME, 'Key': file_name},
                                              ExpiresIn=3600)

        response = rekognition.detect_text(
            Image={'S3Object': {'Bucket': BUCKET_NAME, 'Name': file_name}}
        )

        if 'TextDetections' not in response or not response['TextDetections']:
            return jsonify({'error': 'No text detected'}), 400

        detected_texts = [text['DetectedText'] for text in response['TextDetections']]
        description = " ".join(detected_texts)

        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {'role': 'system', 'content': 'You are a humorous assistant.'},
                {'role': 'user', 'content': f"Generate a very witty dirty minded response based on the following detected text: {description}"}
            ],
            'max_tokens': 50,
            'temperature': 0.7
        }
        openai_response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        app.logger.debug(f"OpenAI response status code: {openai_response.status_code}")
        app.logger.debug(f"OpenAI response content: {openai_response.text}")

        if openai_response.status_code == 429:
            return jsonify({'error': 'OpenAI API quota exceeded. Please check your plan and billing details.'}), 500

        if openai_response.status_code != 200:
            return jsonify({'error': 'Failed to get response from OpenAI'}), 500

        funny_response = openai_response.json()['choices'][0]['message']['content'].strip()
        return jsonify({'funny_response': funny_response})

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
