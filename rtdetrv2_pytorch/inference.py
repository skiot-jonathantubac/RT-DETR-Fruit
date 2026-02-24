import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torchvision.transforms as T
import cv2
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import YAMLConfig

# ── CONFIGURACIÓN ──────────────────────────────────────────
CONFIG    = 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'
WEIGHTS   = 'output/mi_modelo/best.pth'
THRESHOLD = 0.85
CLASSES   = ['apple', 'Healty Apple']
COLORS    = [(0, 0, 255), (0, 255, 0)]  # BGR: rojo, verde
CAMARA    = 0
# ───────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando: {device}")

# Cargar modelo y postprocessor
cfg = YAMLConfig(CONFIG, resume=WEIGHTS)
model = cfg.model
checkpoint = torch.load(WEIGHTS, map_location=device)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

postprocessor = cfg.postprocessor
postprocessor = postprocessor.to(device)
postprocessor.eval()

print("✅ Modelo cargado")

transform = T.Compose([
    T.ToTensor(),
])

# Abrir cámara
cap = cv2.VideoCapture(CAMARA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

print("✅ Cámara abierta — presioná 'Q' para salir, 'S' para guardar screenshot")
print(f"   Threshold inicial: {THRESHOLD} ('+' para subir, '-' para bajar)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el frame")
        break

    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    orig_size = torch.tensor([[h, w]]).to(device)

    # Inferencia + postprocesado oficial
    with torch.no_grad():
        outputs = model(img_tensor, orig_size)
        results = postprocessor(outputs, orig_size)

    labels = results[0]['labels']
    boxes  = results[0]['boxes']
    scores = results[0]['scores']

    # Dibujar detecciones
    detecciones = 0
    for label, box, score in zip(labels, boxes, scores):
        score = score.item()
        label = label.item()

        if score > THRESHOLD:
            detecciones += 1
            bx1, by1, bx2, by2 = [int(v) for v in box.tolist()]

            # Clamp para no salirse del frame
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)

            idx   = label % len(COLORS)
            color = COLORS[idx]
            clase = CLASSES[label] if label < len(CLASSES) else f'clase_{label}'
            texto = f'{clase} {score:.2f}'

            # Caja
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

            # Fondo etiqueta
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (bx1, by1 - th - 8), (bx1 + tw + 4, by1), color, -1)

            # Texto
            cv2.putText(frame, texto, (bx1 + 2, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Info en pantalla
    cv2.putText(frame, f'Detecciones: {detecciones}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f'Threshold: {THRESHOLD:.2f}  |  Q=salir  S=foto  +/-=ajustar', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('RT-DETR - Deteccion en tiempo real', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('screenshot.jpg', frame)
        print("📸 Screenshot guardado!")
    elif key == ord('+'):
        THRESHOLD = min(0.99, round(THRESHOLD + 0.05, 2))
        print(f"Threshold: {THRESHOLD}")
    elif key == ord('-'):
        THRESHOLD = max(0.05, round(THRESHOLD - 0.05, 2))
        print(f"Threshold: {THRESHOLD}")

cap.release()
cv2.destroyAllWindows()
print("👋 Cerrando...")