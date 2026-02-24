import json

for split in ['train', 'valid']:
    path = f'D:/SKIOT/RT-DER_MODELO/Fruit-Detection-1/{split}/_annotations.coco.json'
    
    with open(path) as f:
        data = json.load(f)
    
    total = len(data['annotations'])
    limpias = []
    
    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        # Filtrar cajas con ancho o alto <= 0
        if w > 1 and h > 1:
            limpias.append(ann)
        else:
            print(f"❌ Anotación inválida eliminada: id={ann['id']} bbox={ann['bbox']}")
    
    data['annotations'] = limpias
    print(f"{split}: {total} → {len(limpias)} anotaciones ({total - len(limpias)} eliminadas)")
    
    with open(path, 'w') as f:
        json.dump(data, f)
    
    print(f"✅ {split} guardado\n")