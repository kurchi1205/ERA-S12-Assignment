import torchvision
import numpy as np
from torchvision import transforms
from data import CIFAR10WithAlbumentations
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


means = [0.4914, 0.4822, 0.4465]
stds = [0.2470, 0.2435, 0.2616]

test_transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )
infer_ds = CIFAR10WithAlbumentations('./data', train=False, download=True, transform=test_transforms)
dataloader_args = dict(shuffle=True, batch_size=1)
infer_loader = DataLoader(infer_ds, **dataloader_args)

def plot_misclassified(misclassified):
    f, axarr = plt.subplots(5,2, figsize=(8, 12))
    for num in range(1, 11):
        f.add_subplot(5, 2, num)
        idx = num - 1
        plt.imshow((misclassified[idx]["img"]).astype(np.uint8))
        plt.xlabel(misclassified[idx]["pred_class"], fontsize=15)

    f.tight_layout()
    plt.savefig("misclassified_images.png")
    plt.show()


def get_misclassified_images_with_label(tensor, pred_label, class_to_idx):
    img = get_image(tensor)
    pred_class = ""
    for cls, idx in class_to_idx.items():
        if idx == pred_label:
            pred_class = cls
    return {
        "img": img,
        "pred_class": pred_class,
        "tensor": tensor,
        "pred_idx": pred_label
    }

    
def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)
  

def get_image(tensor):
    img = torchvision.utils.make_grid(tensor)
    img = img.cpu().numpy()
    img = interval_mapping(img, np.min(img), np.max(img), 0, 255)
    unnorm_img = np.transpose(img, (1, 2, 0))
    return unnorm_img

def infer(model, device, infer_loader, misclassified, class_to_idx):
    model.eval()
    with torch.no_grad():
        for data, target in infer_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            if pred.eq(target.view_as(pred)).sum().item() == 0:
                misclassified.append(get_misclassified_images_with_label(data, pred[0][0], class_to_idx))
            if len(misclassified) == 10:
                break