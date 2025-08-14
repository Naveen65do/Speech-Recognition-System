from flask import Flask, render_template, request, redirect, url_for
import os
from stt_whisper_transformers import transcribe_with_whisper

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'audio'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input', methods=['GET', 'POST'])
def input():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return redirect(request.url)
        file = request.files['audio_file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            transcription = transcribe_with_whisper(filepath)
            return render_template('output.html', transcription=transcription)
    return render_template('input.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    os.makedirs('audio', exist_ok=True)
    app.run(debug=True)