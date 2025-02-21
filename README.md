## Identification and Validation of the Molecular Subgroups of Ovarian Cancer and Its Related Cancers using a Self-supervised Machine Learning Method Based on Disulfidptosis-Related Genes

## Requirements
- Python3
- PyTorch (2.4.1)
- torchvision (0.19.1)
- numpy (1.24.4)

## Training
Run ```./train_project/cls_model.py```, training classification models.

```
python ./train_project/cls_model.py  --train_file_path ./labels_data/OV_TCGA.csv --test_file_path ./labels_data/OV_GEO.csv  --best_model_path ./result/best_model --batch_size 64 --lr 0.05 --trail 1
```
Run ```./train_project/generalisation_cls_model.py```, Generalisation and migration experiments on other tumour RNA-seq data.

```
python ./train_project/generalisation_cls_model.py --test_file_path ./labels_data/UCEC_label.csv  --best_model_path ./result/best_model/xxx --save_model_path ./result/UCEC_best_model --batch_size 64 --lr 0.005 --trail 1
```
## Application tools

We produced an offline tool using model weights to assist physicians in subtype discrimination. We created an 64bit application that runs on windows based on the QT platform and the MSVC compiler.
The quoted application source code is located at ``` ./OV_Classification_APP/project/code ```. Open the  ```untitled.pro ``` file with QT to see the entire project.

### Development environment 
- QT (5.15.2)
- MSVC (MSVC2019)
