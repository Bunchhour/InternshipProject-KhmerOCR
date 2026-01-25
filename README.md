# Khmer OCR Project

This project implements Optical Character Recognition (OCR) for Khmer script using deep learning models.

## Project Structure

```
khmer-ocr-project/
│
├── data/                      # Store your datasets here
│   ├── raw/                   # Downloaded datasets (Hanuman, etc.)
│   └── processed/             # Normalized labels and resized images
│
├── checkpoints/               # Where model weights (.pth files) will be saved
│
├── notebooks/                 # For your experiments (Sprints 0 & 1)
│   └── 01_data_exploration.ipynb
│
├── src/                       # Source code (The core logic)
│   ├── __init__.py
│   ├── dataset.py             # Code to load images and labels
│   ├── model.py               # Your CNN, LSTM, and Transformer classes
│   ├── train.py               # The training loop
│   ├── evaluate.py            # Code to calculate CER/WER
│   └── utils.py               # CTC decoding, converting text-to-ID
│
├── .gitignore                 # To ignore data and checkpoints in git
├── README.md                  # Project documentation
└── requirements.txt           # List of libraries to install
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download your dataset and place it in `data/raw/`

3. Start experimenting in the notebooks!

## Usage

### Training
```bash
python src/train.py
```

### Evaluation
```bash
python src/evaluate.py
```

## Metrics

- **CER (Character Error Rate)**: Measures character-level accuracy
- **WER (Word Error Rate)**: Measures word-level accuracy

## License

TBD
