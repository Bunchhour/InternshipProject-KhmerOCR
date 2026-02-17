"""
Evaluation metrics for Khmer OCR
Calculate CER (Character Error Rate) and WER (Word Error Rate)
"""

import editdistance


def calculate_cer(predicted, ground_truth):
    """
    Calculate Character Error Rate
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
    
    Returns:
        Character Error Rate (float)
    """
    distance = editdistance.eval(predicted, ground_truth)
    cer = distance / len(ground_truth) if len(ground_truth) > 0 else 0
    return cer


def calculate_wer(predicted, ground_truth):
    """
    Calculate Word Error Rate
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
    
    Returns:
        Word Error Rate (float)
    """
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    
    distance = editdistance.eval(pred_words, gt_words)
    wer = distance / len(gt_words) if len(gt_words) > 0 else 0
    return wer


def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset"""
    # TODO: Implement full evaluation loop
    pass
