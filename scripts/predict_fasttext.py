#!/usr/bin/env python3
"""
FastText Prediction Script for Document Quality Classification
"""

import fasttext
import argparse
import os

def load_model(model_path):
    """Load the trained FastText model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = fasttext.load_model(model_path)
    return model

def predict_text(model, text, threshold=0.5):
    """Predict quality of a single text"""
    prediction = model.predict(text, k=2)
    
    labels = [label.replace('__label__', '') for label in prediction[0]]
    scores = prediction[1]
    
    primary_label = labels[0]
    primary_score = scores[0]
    
    # Determine if document should be filtered
    is_high_quality = primary_label == 'high' and primary_score >= threshold
    
    return {
        'text': text,
        'predicted_label': primary_label,
        'confidence': primary_score,
        'is_high_quality': is_high_quality,
        'all_predictions': list(zip(labels, scores))
    }

def predict_file(model, input_file, output_file=None, threshold=0.5):
    """Predict quality for all texts in a file"""
    results = []
    
    print(f"Processing file: {input_file}")
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            # Extract true label if present (for evaluation)
            true_label = None
            if line.startswith('__label__'):
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    true_label = parts[0].replace('__label__', '')
                    text = parts[1]  # Use clean text for display
                else:
                    text = line
            else:
                text = line
                
            result = predict_text(model, line, threshold)  # FastText handles full line with labels
            result['line_number'] = line_num
            result['true_label'] = true_label
            result['display_text'] = text  # Clean text for display
            results.append(result)
    
    # Calculate metrics if true labels are available
    labeled_results = [r for r in results if r.get('true_label')]
    
    if labeled_results:
        # Calculate confusion matrix
        tp = sum(1 for r in labeled_results if r['true_label'] == 'high' and r['predicted_label'] == 'high')
        fp = sum(1 for r in labeled_results if r['true_label'] == 'low' and r['predicted_label'] == 'high')
        tn = sum(1 for r in labeled_results if r['true_label'] == 'low' and r['predicted_label'] == 'low')
        fn = sum(1 for r in labeled_results if r['true_label'] == 'high' and r['predicted_label'] == 'low')
        
        # Calculate metrics
        accuracy = (tp + tn) / len(labeled_results)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nEvaluation Metrics:")
        print("=" * 50)
        print(f"Accuracy:  {accuracy:.4f} ({(tp + tn)}/{len(labeled_results)})")
        print(f"Precision: {precision:.4f} (TP={tp}, FP={fp})")
        print(f"Recall:    {recall:.4f} (TP={tp}, FN={fn})")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              High   Low")
        print(f"Actual High   {tp:4d}  {fn:4d}")
        print(f"       Low    {fp:4d}  {tn:4d}")
    
    # Print sample results
    print(f"\nSample Results (first 5):")
    print("=" * 80)
    
    high_quality_count = 0
    for i, result in enumerate(results[:5]):
        quality_status = "✓ HIGH QUALITY" if result['is_high_quality'] else "✗ LOW QUALITY"
        high_quality_count += result['is_high_quality']
        
        # Show accuracy marker if true label available
        accuracy_marker = ""
        if result.get('true_label'):
            correct = result['predicted_label'] == result['true_label']
            accuracy_marker = f" {'✓' if correct else '✗'} (True: {result['true_label']})"
        
        print(f"Line {result['line_number']}: {quality_status}{accuracy_marker}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        # Use clean display text
        display_text = result.get('display_text', result['text'])
        print(f"Text: {display_text[:100]}...")
        print("-" * 40)
    
    # Summary
    total_high_quality = sum(1 for r in results if r['is_high_quality'])
    print(f"\nOverall Summary:")
    print(f"Total documents: {len(results)}")
    print(f"Predicted high quality: {total_high_quality}")
    print(f"Predicted low quality: {len(results) - total_high_quality}")
    print(f"Quality rate: {total_high_quality/len(results)*100:.1f}%")
    
    # Save to output file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write("line_number,predicted_label,confidence,is_high_quality,text\n")
            for result in results:
                text_escaped = result['text'].replace('"', '""')
                f.write(f"{result['line_number']},{result['predicted_label']},{result['confidence']:.4f},{result['is_high_quality']},\"{text_escaped}\"\n")
        print(f"Results saved to: {output_file}")
    
    return results

def filter_documents(model, input_file, output_file, threshold=0.5):
    """Filter documents keeping only high quality ones"""
    print(f"Filtering documents from: {input_file}")
    print(f"Saving high quality documents to: {output_file}")
    
    kept_count = 0
    total_count = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            text = line.strip()
            if not text:
                continue
                
            total_count += 1
            result = predict_text(model, text, threshold)
            
            if result['is_high_quality']:
                outfile.write(line)
                kept_count += 1
    
    print(f"Filtering complete!")
    print(f"Original documents: {total_count}")
    print(f"High quality kept: {kept_count}")
    print(f"Low quality filtered: {total_count - kept_count}")
    print(f"Retention rate: {kept_count/total_count*100:.1f}%")

def interactive_mode(model, threshold=0.5):
    """Interactive mode for testing individual texts"""
    print("Interactive mode - Enter text to classify (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
                
            if not text:
                continue
                
            result = predict_text(model, text, threshold)
            quality_status = "HIGH QUALITY ✓" if result['is_high_quality'] else "LOW QUALITY ✗"
            
            print(f"\nPrediction: {quality_status}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"All predictions: {result['all_predictions']}")
            
        except KeyboardInterrupt:
            break
    
    print("\nExiting interactive mode.")

def main():
    parser = argparse.ArgumentParser(description='Predict document quality using trained FastText model')
    parser.add_argument('--model', default='document_quality_model.bin', help='Path to trained model')
    parser.add_argument('--input', help='Input file to classify')
    parser.add_argument('--output', help='Output file for results (CSV format)')
    parser.add_argument('--filter', help='Output file for filtered high-quality documents')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for high quality (default: 0.5)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--text', help='Single text to classify')
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've trained a model first using: python train_fasttext.py")
        return
    
    # Single text prediction
    if args.text:
        result = predict_text(model, args.text, args.threshold)
        quality_status = "HIGH QUALITY ✓" if result['is_high_quality'] else "LOW QUALITY ✗"
        print(f"Text: {args.text}")
        print(f"Prediction: {quality_status}")
        print(f"Confidence: {result['confidence']:.4f}")
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, args.threshold)
        return
    
    # File processing
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found!")
            return
            
        # Prediction mode
        predict_file(model, args.input, args.output, args.threshold)
        
        # Filtering mode
        if args.filter:
            filter_documents(model, args.input, args.filter, args.threshold)
    else:
        print("No input specified. Use --input for file processing, --text for single prediction, or --interactive for interactive mode")
        print("Example usage:")
        print("  python predict_fasttext.py --input documents.txt --output results.csv")
        print("  python predict_fasttext.py --text 'This is a sample document to classify'")
        print("  python predict_fasttext.py --interactive")

if __name__ == "__main__":
    main()