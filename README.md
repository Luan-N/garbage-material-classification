# Garbage Material Classification Using Convolutional Neural Networks

## Team Members
- Luan Nguyen
- Nurislam Saliev

## Overview
This project classifies garbage materials from images using deep learning. We will compare:
- a custom CNN
- a pre-trained MobileNetV2 model using transfer learning

## Dataset
We use the **RealWaste Image Classification Dataset** from Kaggle:  
https://www.kaggle.com/datasets/joebeachcapital/realwaste

The dataset contains about **4,753 images** across **9 classes**:
- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

## Project Structure
- `data/` dataset instructions only
- `src/` preprocessing, training, and evaluation code
- `results/` graphs, metrics, and model comparison outputs

## Setup
1. Clone the repository.
2. Install dependencies with:
`python -m venv .venv`
`source .venv/bin/activate`
`pip install -r requirements.txt`
3. Download the dataset from Kaggle.
4. Extract it into the `data/` folder like this:

```text
data/
└── RealWaste/
    ├── Cardboard/
    ├── Food Organics/
    ├── Glass/
    ├── Metal/
    ├── Miscellaneous Trash/
    ├── Paper/
    ├── Plastic/
    ├── Textile Trash/
    └── Vegetation/
```