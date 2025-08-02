# app.py
from flask import Flask, render_template, Response, jsonify
from blink_detector import generate_frames, blink_count, stop

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_blink_count')
def get_blink_count():
    return jsonify({'count': blink_count})

@app.route('/stop', methods=['POST'])
def stop_video():
    stop()
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(debug=True)
