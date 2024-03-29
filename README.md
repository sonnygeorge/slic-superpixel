
## 1. SLIC

By default, SKImage enforces connectivity between superpixels. This leads to a much more coherent result for images that are noisy (e.g. handles leaves better). The major drawback of my algorithm is that I did not implement this. However, when I toggle the `enforce_connectivity` flag in the SKImage implementation, the results are comparable to that of my algorithm.

### `compactness=10`

#### Mine

![img](data/superpixels_10.png)

#### Skimage `enforce_connectivity=False`

![img](data/superpixels_skimage_10.png)

#### Skimage `enforce_connectivity=True`

![img](data/superpixels_skimage_10_connected.png)

### `compactness=30`

#### Mine

![img](data/superpixels_30.png)

#### Skimage `enforce_connectivity=False`

![img](data/superpixels_skimage_30.png)

#### Skimage `enforce_connectivity=True`

![img](data/superpixels_skimage_30_connected.png)

### `compactness=50`

#### Mine

![img](data/superpixels_50.png)

#### Skimage `enforce_connectivity=False`

![img](data/superpixels_skimage_50.png)

#### Skimage `enforce_connectivity=True`

![img](data/superpixels_skimage_50_connected.png)

## 2. Grad-CAM

### Demo 1

```bash
python main.py demo1 -i shrek.jpeg -a resnet152 -t layer4
```

| Predicted Class | #1 mask | #3 neck brace |
| - | - | - |
| Grad-CAM | ![img](data/0-resnet152-gradcam-layer4-mask.png) | ![img](data/0-resnet152-gradcam-layer4-neck_brace.png) |
| Vanilla backpropagation | ![img](data/0-resnet152-vanilla-mask.png) | ![img](data/0-resnet152-vanilla-neck_brace.png) |
| Guided Grad-CAM | ![img](data/0-resnet152-guided_gradcam-layer4-mask.png) | ![img](data/0-resnet152-guided_gradcam-layer4-neck_brace.png) |

### Demo 2

```
python main.py demo2 -i bull_mastiff.jpg  
python main.py demo2 -i perturbed.jpg  
```

| Layer | `relu` | `layer1` | `layer2` | `layer3` | `layer4` |
| - | - | - | - | - | - |
| Grad-CAM | ![img](data/0-resnet152-gradcam-relu-bull_mastiff.png) | ![img](data/0-resnet152-gradcam-layer1-bull_mastiff.png) | ![img](data/0-resnet152-gradcam-layer2-bull_mastiff.png) | ![img](data/0-resnet152-gradcam-layer3-bull_mastiff.png) | ![img](data/0-resnet152-gradcam-layer4-bull_mastiff.png) |
| Grad-CAM (After Perturbation) | ![img](data/0-resnet152-gradcam-relu-bull_mastiff_perturbed.png) | ![img](data/0-resnet152-gradcam-layer1-bull_mastiff_perturbed.png) | ![img](data/0-resnet152-gradcam-layer2-bull_mastiff_perturbed.png) | ![img](data/0-resnet152-gradcam-layer3-bull_mastiff_perturbed.png) | ![img](data/0-resnet152-gradcam-layer4-bull_mastiff_perturbed.png) |
