
import os
import pandas as pd
import torch
import torch.nn.functional as F
from dataload import get_loaders, CFG
from train import BaseModel

def inference_tta():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data loader and class names
    _, _, test_loader, class_names = get_loaders()

    # Load model
    model = BaseModel(num_classes=len(class_names))
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for images in test_loader:
            # images: tuple if coming from Dataset returning tuple; ensure tensor
            if isinstance(images, (list, tuple)):
                images = images[0]

            images = images.to(device)  # (B, 3, H, W)

            # TTA 1: original
            outputs_orig = model(images)
            probs_orig = F.softmax(outputs_orig, dim=1)

            # TTA 2: horizontal flip
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            probs_flip = F.softmax(outputs_flip, dim=1)

            # Average TTA results
            probs_avg = (probs_orig + probs_flip) * 0.5  # (B, num_classes)

            # Accumulate results
            for prob in probs_avg.cpu():
                result = { class_names[i]: prob[i].item() for i in range(len(class_names)) }
                results.append(result)

    # Construct DataFrame and save
    pred = pd.DataFrame(results)
    # Extract IDs from filenames
    ids = []
    for sample in test_loader.dataset.samples:
        img_path = sample[0]
        id_str = os.path.basename(img_path).rsplit('.', 1)[0]
        ids.append(id_str)
    pred['ID'] = ids
    pred = pred[['ID'] + class_names]
    pred.to_csv('submission_tta.csv', index=False, encoding='utf-8-sig')
    print("Saved TTA predictions to submission_tta.csv")

if __name__ == "__main__":
    inference_tta()
