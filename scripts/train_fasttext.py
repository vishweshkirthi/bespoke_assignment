#!/usr/bin/env python3
"""
FastText Training Script for Document Quality Classification
"""

import fasttext
import os
import argparse
import random

def train_model(train_file, val_file, model_output):
    """Train FastText model for document quality classification"""
    
    print(f"Training FastText model...")
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}")
    
    # Train the model
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.1,          # learning rate
        epoch=25,        # number of epochs
        wordNgrams=2,    # use bigrams
        dim=100,         # dimension of word vectors
        ws=5,            # window size
        minCount=1,      # minimum word count
        minn=3,          # min char ngram
        maxn=6,          # max char ngram
        neg=5,           # negative sampling
        loss='softmax',  # loss function
        verbose=2        # verbose output
    )
    
    # Save the model
    model.save_model(model_output)
    print(f"Model saved to: {model_output}")
    
    # Test on validation set
    if os.path.exists(val_file):
        print(f"\nEvaluating on validation set...")
        result = model.test(val_file)
        print(f"Number of examples: {result[0]}")
        print(f"Precision: {result[1]:.4f}")
        print(f"Recall: {result[2]:.4f}")
        
        # Test individual predictions
        print(f"\nSample predictions:")
        with open(val_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 examples
                    break
                text = line.strip().split(' ', 1)[1] if ' ' in line else line.strip()
                prediction = model.predict(text, k=2)
                labels = [label.replace('__label__', '') for label in prediction[0]]
                scores = prediction[1]
                print(f"Text: {text[:60]}...")
                print(f"Predicted: {labels[0]} (confidence: {scores[0]:.4f})")
                print(f"Alternative: {labels[1]} (confidence: {scores[1]:.4f})")
                print("-" * 50)
    
    return model

def combine_datasets(positive_file, negative_file, output_file, train_split=0.8):
    """Combine positive and negative files into train/val splits"""
    
    print(f"Reading positive examples from: {positive_file}")
    with open(positive_file, 'r', encoding='utf-8') as f:
        positive_lines = f.readlines()
    
    print(f"Reading negative examples from: {negative_file}")
    with open(negative_file, 'r', encoding='utf-8') as f:
        negative_lines = f.readlines()
    
    print(f"Positive examples: {len(positive_lines)}")
    print(f"Negative examples: {len(negative_lines)}")
    
    # Combine and shuffle
    all_lines = positive_lines + negative_lines
    random.shuffle(all_lines)
    
    # Split into train/val
    split_idx = int(len(all_lines) * train_split)
    train_lines = all_lines[:split_idx]
    val_lines = all_lines[split_idx:]
    
    # Write training file
    train_file = output_file
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # Write validation file
    val_file = output_file.replace('.txt', '_val.txt')
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"Training examples: {len(train_lines)}")
    print(f"Validation examples: {len(val_lines)}")
    print(f"Training file: {train_file}")
    print(f"Validation file: {val_file}")
    
    return train_file, val_file

def main():
    parser = argparse.ArgumentParser(description='Train FastText model for document quality classification')
    parser.add_argument('--train', default='train.txt', help='Training data file')
    parser.add_argument('--val', default='val.txt', help='Validation data file')
    parser.add_argument('--output', default='document_quality_model.bin', help='Output model file')
    parser.add_argument('--positive', help='Positive examples file (will combine with negatives)')
    parser.add_argument('--negative', default='data/train_negative.txt', help='Negative examples file')
    parser.add_argument('--combined_output', default='combined_train.txt', help='Combined training file output')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # If positive file is provided, combine datasets first
    if args.positive:
        if not os.path.exists(args.positive):
            print(f"Error: Positive file {args.positive} not found!")
            return
        
        if not os.path.exists(args.negative):
            print(f"Error: Negative file {args.negative} not found!")
            return
        
        # Combine datasets
        train_file, val_file = combine_datasets(
            args.positive, 
            args.negative, 
            args.combined_output
        )
    else:
        # Use existing train/val files
        train_file = args.train
        val_file = args.val
        
        if not os.path.exists(train_file):
            print(f"Error: Training file {train_file} not found!")
            return
        
        if not os.path.exists(val_file):
            print(f"Warning: Validation file {val_file} not found!")
    
    # Train the model
    model = train_model(train_file, val_file, args.output)
    
    print(f"\nTraining completed successfully!")
    print(f"You can now use the model with: python predict_fasttext.py")

if __name__ == "__main__":
    main()