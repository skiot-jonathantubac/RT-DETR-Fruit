import json

with open('D:/SKIOT/RT-DER_MODELO/Fruit-Detection-1/train/_annotations.coco.json') as f:
    data = json.load(f)

print("Categorias:", data['categories'])
ids = [ann['category_id'] for ann in data['annotations']]
print("IDs únicos encontrados:", sorted(set(ids)))
print("Max ID:", max(ids))