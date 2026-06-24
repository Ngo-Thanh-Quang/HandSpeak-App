from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from analyze_video import process_and_compare

app = Flask(__name__)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'E',
    6: 'F',
    7: 'G',
    8: 'H',
    9: 'I',
    10: 'J',
    11: 'K',
    12: 'L',
    13: 'M',
    14: 'N',
    15: 'O',
    16: 'P',
    17: 'Q',
    18: 'R',
    19: 'S',
    20: 'T',
    21: 'U',
    22: 'V',
    23: 'W',
    24: 'X',
    25: 'Y',
    26: 'Z',
    27: 'Space',
    28: 'Thanks ',
    29: 'Hello ',
    30: 'Me ',
    31: 'Help ',
    32: 'Name ',
    33: 'My ',
    34: 'Is ',
    35: 'What ',
    36: 'You ',
    37: 'Fine ',
    38: 'Yes ',
    39: 'No ',
    40: 'Sorry ',
    41: 'Understand ',
    42: 'Again ',
    43: 'Ready ',
    44: 'Great ',
    45: 'Friend ',
    46: 'Not Yet ',
    47: 'How ',
    48: 'Your ',
}

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

def extract_hand_features(results):
    if not results.multi_hand_landmarks:
        return None

    detected_hands = {}
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label
        landmarks = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
        detected_hands[label] = landmarks

    all_points = [point for landmarks in detected_hands.values() for point in landmarks]
    min_x = min(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_x = max(x for x, _ in all_points)
    max_y = max(y for _, y in all_points)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)

    data_aux = []
    for hand_label in ('Left', 'Right'):
        landmarks = detected_hands.get(hand_label)
        if landmarks:
            for x, y in landmarks:
                data_aux.append((x - min_x) / width)
                data_aux.append((y - min_y) / height)
        else:
            data_aux.extend([0] * 42)

    return data_aux

@app.route('/', methods=['GET'])
def home():
    return "Server is running...", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Không có hình ảnh nào'}), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Không đọc được hình ảnh'}), 400

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = extract_hand_features(results)
    if data_aux:
        prediction = model.predict([np.asarray(data_aux)])
        predicted_label = int(prediction[0])
        predicted_character = labels_dict.get(predicted_label, 'UNKNOWN {}'.format(predicted_label))

        return jsonify(predicted_character)

    return jsonify("No hand detection!"), 200  # Trả về chuỗi rỗng nếu không nhận diện được

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files or 'gif_url' not in request.form:
        return jsonify({'error': 'Missing data'}), 400

    video = request.files['video']
    gif_url = request.form['gif_url']

    # Tạo thư mục temp nếu chưa có
    os.makedirs('temp', exist_ok=True)

    save_path = os.path.join("temp", video.filename)
    video.save(save_path)

    # Gọi hàm xử lý AI
    score = process_and_compare(save_path, gif_url)

    return jsonify({'score': round(score, 2)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
