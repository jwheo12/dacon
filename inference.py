
import os
import pandas as pd
import torch
import torch.nn.functional as F
from dataload import get_loaders
from train import BaseModel

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get loaders and class names
    _, _, test_loader, class_names = get_loaders()
    
    # Load model
    model = BaseModel(num_classes=len(class_names))
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in test_loader:
            # Ensure images tensor extracted
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)  # (B, 3, H, W)

            # TTA: original
            outputs_orig = model(images)
            probs_orig = F.softmax(outputs_orig, dim=1)

            # TTA: horizontal flip
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            probs_flip = F.softmax(outputs_flip, dim=1)

            # Average
            probs_avg = (probs_orig + probs_flip) * 0.5

            # Collect
            for prob in probs_avg.cpu():
                results.append({class_names[i]: prob[i].item() for i in range(len(class_names))})

    # Create DataFrame
    pred = pd.DataFrame(results)
    # Extract IDs
    ids = [os.path.basename(path).rsplit('.', 1)[0] for path, *_ in test_loader.dataset.samples]
    pred['ID'] = ids
    pred = pred[['ID'] + class_names]
    # Save
    pred.to_csv('submission_tta.csv', index=False, encoding='utf-8-sig')
    print("Saved submission_tta.csv")

if __name__ == "__main__":
    main()
