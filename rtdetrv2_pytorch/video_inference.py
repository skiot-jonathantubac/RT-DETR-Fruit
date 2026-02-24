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
VIDEO     = 'tests/test_video.mp4'        # ← ruta a tu video
SALIDA    = 'tests/output/test_resultado.mp4'    # ← video de salida con detecciones
THRESHOLD = 0.85
CLASSES   = ['apple', 'Healty Apple']
COLORS    = [(0, 0, 255), (0, 255, 0)]  # BGR: rojo, verde
GUARDAR   = True               # ← False si solo querés ver sin guardar
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

postprocessor = cfg.postprocessor.to(device)
postprocessor.eval()
print("✅ Modelo cargado")

transform = T.Compose([T.ToTensor()])

# Abrir video
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"❌ No se pudo abrir el video: {VIDEO}")
    exit()

# Info del video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✅ Video: {w}x{h} | {fps:.1f} FPS | {total_frames} frames")

# Configurar escritor de video de salida
writer = None
if GUARDAR:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(SALIDA, fourcc, fps, (w, h))
    print(f"✅ Guardando resultado en: {SALIDA}")

print("▶️  Procesando... presioná 'Q' para salir, ESPACIO para pausar")

frame_idx = 0
pausado   = False

while True:
    if not pausado:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video terminado")
            break
        frame_idx += 1

    # Preprocesar
    img_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor  = transform(img_resized).unsqueeze(0).to(device)
    orig_size   = torch.tensor([[h, w]]).to(device)

    # Inferencia
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
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)

            idx   = label % len(COLORS)
            color = COLORS[idx]
            clase = CLASSES[label] if label < len(CLASSES) else f'clase_{label}'
            texto = f'{clase} {score:.2f}'

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (bx1, by1 - th - 8), (bx1 + tw + 4, by1), color, -1)
            cv2.putText(frame, texto, (bx1 + 2, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Info en pantalla
    progreso = f'Frame: {frame_idx}/{total_frames} ({100*frame_idx//total_frames}%)'
    cv2.putText(frame, progreso, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f'Detecciones: {detecciones}  |  Threshold: {THRESHOLD:.2f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if pausado:
        cv2.putText(frame, '⏸ PAUSADO', (w//2 - 80, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Guardar frame
    if GUARDAR and writer:
        writer.write(frame)

    cv2.imshow('RT-DETR - Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("⏹ Detenido por usuario")
        break
    elif key == ord(' '):
        pausado = not pausado
        print("⏸ Pausado" if pausado else "▶️  Reanudado")
    elif key == ord('s'):
        cv2.imwrite(f'frame_{frame_idx}.jpg', frame)
        print(f"📸 Frame {frame_idx} guardado!")
    elif key == ord('+'):
        THRESHOLD = min(0.99, round(THRESHOLD + 0.05, 2))
        print(f"Threshold: {THRESHOLD}")
    elif key == ord('-'):
        THRESHOLD = max(0.05, round(THRESHOLD - 0.05, 2))
        print(f"Threshold: {THRESHOLD}")

# Liberar recursos
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print(f"👋 Listo! Frames procesados: {frame_idx}/{total_frames}")