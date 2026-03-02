import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
with open('D:/SKIOT/RT-DER_MODELO/Fruit-Detection-1/train/_annotations.coco.json') as f:
    data = json.load(f)
clases = [c['name'] for c in sorted(data['categories'], key=lambda x: x['id'])]
print(clases)

from src.core import YAMLConfig

# ── CONFIGURACIÓN ──────────────────────────────────────────
CONFIG    = 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'
WEIGHTS   = 'models/Frutas/last.pth'
THRESHOLD = 0.85
CLASSES = ['Grocery-Items', 'Apple', 'Banana', 'Green Pepper', 'Lemon', 'Orange', 'Red Pepper', 'Strawberry', 'Tomato']
COLORS = [
    (0,   0,   255),  # rojo
    (0,   255, 0  ),  # verde
    (255, 0,   0  ),  # azul
    (0,   255, 255),  # amarillo
    (255, 0,   255),  # magenta
    (255, 165, 0  ),  # naranja
    (128, 0,   128),  # morado
    (0,   128, 128),  # verde oscuro
    (255, 192, 203),  # rosa
]
CAMARA    = 0
INPUT_SIZE = 640
# ───────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando: {device}")

# ── Cargar modelo ──────────────────────────────────────────
cfg = YAMLConfig(CONFIG, resume=WEIGHTS)

model = cfg.model
checkpoint = torch.load(WEIGHTS, map_location=device)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

postprocessor = cfg.postprocessor.to(device)
postprocessor.eval()

print("✅ Modelo cargado")


# ── LETTERBOX (mantener proporción) ───────────────────────
def letterbox(im, new_size=640):
    h, w = im.shape[:2]
    scale = new_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(im, (new_w, new_h))

    canvas = np.zeros((new_size, new_size, 3), dtype=np.uint8)

    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2

    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    return canvas, scale, pad_x, pad_y


# ── Abrir cámara ───────────────────────────────────────────
cap = cv2.VideoCapture(CAMARA)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

print("✅ Cámara abierta — Q para salir | +/- para threshold")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🔥 Letterbox correcto
    img_letterbox, scale, pad_x, pad_y = letterbox(img_rgb, INPUT_SIZE)

    img_tensor = torch.from_numpy(img_letterbox).float()
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    orig_size = torch.tensor([[INPUT_SIZE, INPUT_SIZE]]).to(device)

    # ── Inferencia ──────────────────────────────────────────
    with torch.no_grad():
        outputs = model(img_tensor, orig_size)
        results = postprocessor(outputs, orig_size)

    labels = results[0]['labels']
    boxes  = results[0]['boxes']
    scores = results[0]['scores']

    detecciones = 0

    for label, box, score in zip(labels, boxes, scores):
        score = score.item()
        label = label.item()

        if score > THRESHOLD:
            detecciones += 1

            x1, y1, x2, y2 = box.tolist()

            # 🔥 Quitar padding y reescalar a tamaño original
            x1 = (x1 - pad_x) / scale
            x2 = (x2 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            y2 = (y2 - pad_y) / scale

            x1 = int(max(0, min(orig_w, x1)))
            x2 = int(max(0, min(orig_w, x2)))
            y1 = int(max(0, min(orig_h, y1)))
            y2 = int(max(0, min(orig_h, y2)))

            idx   = label % len(COLORS)
            color = COLORS[idx]
            clase = CLASSES[label] if label < len(CLASSES) else f'class_{label}'
            texto = f'{clase} {score:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

            cv2.putText(frame, texto, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f'Detecciones: {detecciones}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("RT-DETRv2 - Tiempo Real", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        THRESHOLD = min(0.99, round(THRESHOLD + 0.05, 2))
        print("Threshold:", THRESHOLD)
    elif key == ord('-'):
        THRESHOLD = max(0.05, round(THRESHOLD - 0.05, 2))
        print("Threshold:", THRESHOLD)

cap.release()
cv2.destroyAllWindows()
print("👋 Cerrando...")