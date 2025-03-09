import os
import cv2
import numpy as np


def get_features(train_folder):
    features = []
    responses = []

    for folder_name in os.listdir(train_folder):
        folder_path = os.path.join(train_folder, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                img = cv2.imread(file_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                img_resized = cv2.resize(thresh, (120, 120), interpolation=cv2.INTER_AREA)
                features.append(img_resized)

                if len(folder_name) > 1:
                    responses.append(ord(folder_name[1]))
                else:
                    responses.append(ord(folder_name[0]))

    features = np.array(features)
    responses = np.array(responses, dtype=np.float32)

    return features, responses

def recognize(image_path, knn_model):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    recognized_text = ""
    last_x = 0
    space_threshold = 30
    i_dot_threshold = 30

    processed_contours = []

    for i, contour in enumerate(contours):
        if i in processed_contours:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if x - last_x > space_threshold and last_x != 0:
            recognized_text += " "

        if cv2.contourArea(contour) < 300:
            for j in range(0, len(contours)):
                if j in processed_contours or j == i:
                    continue

                x2, y2, w2, h2 = cv2.boundingRect(contours[j])
                if abs(x - x2) < i_dot_threshold:
                    if i > j:
                        recognized_text = recognized_text[:-1] + "i"
                    else:
                        recognized_text += "i"

                    processed_contours.append(j)
                    last_x = x + w
                    break
        else:
            x -= 3
            y -= 3
            w += 6
            h += 6
            x = max(0, x)
            y = max(0, y)

            char_img = thresh[y:y + h, x:x + w]
            char_img_resized = cv2.resize(char_img, (120, 120), interpolation=cv2.INTER_AREA)
            sample = char_img_resized.reshape(1, -1).astype(np.float32)
            retval, results, neigh_resp, dists = knn_model.findNearest(sample, k=3)
            predicted_label = chr(int(results[0][0]))
            recognized_text += predicted_label
            last_x = x + w

    return recognized_text, img


train_folder = 'task/train'
features, responses = get_features(train_folder)
knn = cv2.ml.KNearest_create()
samples = features.reshape(features.shape[0], -1).astype(np.float32)
knn.train(samples, cv2.ml.ROW_SAMPLE, responses)

image_to_recognize = 'task/0.png'
recognized_text, output_image = recognize(image_to_recognize, knn)
print(f"Распознанный текст: {recognized_text}")

image_to_recognize = 'task/1.png'
recognized_text, output_image2 = recognize(image_to_recognize, knn)
print(f"Распознанный текст: {recognized_text}")

image_to_recognize = 'task/2.png'
recognized_text, output_image = recognize(image_to_recognize, knn)
print(f"Распознанный текст: {recognized_text}")

image_to_recognize = 'task/3.png'
recognized_text, output_image = recognize(image_to_recognize, knn)
print(f"Распознанный текст: {recognized_text}")

image_to_recognize = 'task/4.png'
recognized_text, output_image = recognize(image_to_recognize, knn)
print(f"Распознанный текст: {recognized_text}")

image_to_recognize = 'task/5.png'
recognized_text, output_image = recognize(image_to_recognize, knn)
print(f"Распознанный текст: {recognized_text}")

image_to_recognize = 'task/6.png'
recognized_text, output_image = recognize(image_to_recognize, knn)
print(f"Распознанный текст: {recognized_text}")
