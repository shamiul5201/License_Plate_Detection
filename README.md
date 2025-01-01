# Vehicle Registration Plate Detection using Detectron2

This project implements a vehicle registration plate detection system using Detectron2, a powerful object detection and segmentation framework built on PyTorch. The project involves training a model on a custom dataset to identify and localize registration plates in images.

![license_plate_detection](https://github.com/user-attachments/assets/b805bc3e-d6d8-460a-bdf4-1fb2ea189d63)

## Installation

1. Install Detectron2 and other dependencies:

```bash
!pip install pycocotools
```

2. Set up the Detectron2 environment:

```python
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
```

3. Import necessary libraries:

```python
import numpy as np
import cv2, random, os
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
```

---

## Dataset Preparation

1. **Directory Structure**
   - Ensure the dataset is organized as follows:
     ```
     /Dataset
     |-- train
     |   |-- Vehicle registration plate
     |       |-- Images
     |       |-- Label
     |-- validation
         |-- Vehicle registration plate
             |-- Images
             |-- Label
     ```

2. **Annotations**
   - Annotations are stored in `.txt` files, with bounding box coordinates for each image.
   - Example of an annotation line:
     ```
     class_id x_center y_center width height xmin ymin xmax ymax
     ```

3. **Dataset Registration**
   - Register the datasets for training and validation using the following code:

```python
DatasetCatalog.register(name='train_regn_plate', func=lambda: get_regn_plate_dicts(data_root, train_txt))
DatasetCatalog.register(name='val_regn_plate', func=lambda: get_regn_plate_dicts(data_root, val_txt))
MetadataCatalog.get('train_regn_plate').set(thing_classes=['regn_plate'])
MetadataCatalog.get('val_regn_plate').set(thing_classes=['regn_plate'])
```

---

## Code Overview

### Dataset Registration

The function `get_regn_plate_dicts` processes the annotations and image metadata into a Detectron2-compatible format. It:
- Reads image dimensions and paths.
- Parses bounding box annotations.
- Outputs a list of dictionaries containing image and annotation data.

### Visualization

Visualize random samples from the dataset to verify correct registration:

```python
for d in random.sample(val_data_dict, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=val_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.figure(figsize=(12, 12))
    plt.imshow(vis.get_image())
    plt.show()
```

### Training

1. Configure the model using a pre-trained RetinaNet from the Detectron2 model zoo.
2. Customize solver parameters and dataset settings:

```python
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_50_FPN_3x.yaml'))
cfg.DATASETS.TRAIN = ('train_regn_plate',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/retinanet_R_50_FPN_3x.yaml')
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = max_iter
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.OUTPUT_DIR = 'outputs'
```

3. Train the model:

```python
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

### Inference

Perform predictions on validation data and visualize the results:

```python
predictor = DefaultPredictor(cfg)
for d in random.sample(val_data_dict, 5):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=val_metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    plt.figure(figsize=(12, 12))
    plt.imshow(v.get_image())
    plt.show()
```

---

## Evaluation

1. Evaluate the model on the validation dataset using COCO metrics:

```python
evaluator = COCOEvaluator(dataset_name='val_regn_plate', output_dir=eval_dir, distributed=False)
val_loader = build_detection_test_loader(cfg, 'val_regn_plate')
inference_on_dataset(trainer.model, val_loader, evaluator)
```

## Acknowledgments

- **Detectron2**: A modular and scalable object detection library.
- **Dataset**: Custom dataset for vehicle registration plate detection.

