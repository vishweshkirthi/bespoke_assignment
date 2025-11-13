#!/usr/bin/env python3
"""
Create train/val/test datasets from news and common crawl data
"""

import pandas as pd
import json
import random
import os
import re
import unicodedata

def sample_news_data(parquet_file, output_dir, total_samples=35000):
    """Sample news data from parquet file"""
    print(f"Reading news data from: {parquet_file}")
    
    # Read parquet file
    df = pd.read_parquet(parquet_file)
    df = df[df['language'] == 'en']
    print(f"Total news articles: {len(df)}")
    
    # Sample random articles
    if len(df) > total_samples:
        df_sampled = df.sample(n=total_samples, random_state=42)
    else:
        df_sampled = df
        print(f"Warning: Only {len(df)} articles available, using all")
    
    # Extract text (adjust column names as needed)
    texts = []
    for _, row in df_sampled.iterrows():
        # Try different possible column names for text content
        text = ""
        for col in ['plain_text', 'text', 'content', 'body', 'article', 'description']:
            if col in row and pd.notna(row[col]):
                text = str(row[col]).strip()
                break
        
        # Try title + text combination
        if not text and 'title' in row:
            text = str(row['title']).strip()
        
        if text and len(text) > 50:  # Basic quality filter
            # Clean text
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = ' '.join(text.split())  # Remove extra spaces
            texts.append(text)
    
    print(f"Extracted {len(texts)} valid news articles")
    
    # Split into train/val/test (adjusted for available data)
    random.shuffle(texts)
    train_texts = texts[:20000]  # 20k for training
    val_texts = texts[20000:30000]  # 10k for validation
    test_texts = texts[30000:]  # ~10k for test
    
    print(f"News split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Write files
    with open(f"{output_dir}/train_positive.txt", 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(f"__label__high {text}\n")
    
    return val_texts, test_texts

def sample_crawl_data(jsonl_file, output_dir, total_samples=35000):
    """Sample common crawl data from JSONL file"""
    print(f"Reading common crawl data from: {jsonl_file}")
    
    texts = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0:
                print(f"Processed {line_num} lines, collected {len(texts)} valid texts")
            if len(texts) >= total_samples:
                break
                
            try:
                data = json.loads(line.strip())
                
                # Try different possible keys for text content
                text = ""
                for key in ['text', 'content', 'body', 'raw_content', 'page_content']:
                    if key in data and data[key]:
                        text = str(data[key]).strip()
                        break
                
                if text and len(text) > 50:  # Basic quality filter
                    # Clean text
                    text = text.replace('\n', ' ').replace('\r', ' ')
                    text = ' '.join(text.split())  # Remove extra spaces
                    
                    # Additional quality filters for common crawl
                    if len(text) < 2000 and len(text.split()) > 10:  # Not too long, has words
                        texts.append(text)
                        
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
    
    print(f"Extracted {len(texts)} valid common crawl texts")
    
    # Shuffle and split (adjusted for available data)
    random.shuffle(texts)
    train_texts = texts[:20000]  # 20k for training
    val_texts = texts[20000:30000]  # 10k for validation
    test_texts = texts[30000:]  # 10k for test
    
    print(f"Crawl split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Write training file
    with open(f"{output_dir}/train_negative.txt", 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(f"__label__low {text}\n")
    
    return val_texts, test_texts

def create_combined_files(news_val, news_test, crawl_val, crawl_test, output_dir):
    """Create combined validation and test files"""
    
    # Ensure balanced splits - take equal numbers from each
    min_val_size = min(len(news_val), len(crawl_val))
    min_test_size = min(len(news_test), len(crawl_test))
    
    print(f"Balancing validation: taking {min_val_size} from each class")
    print(f"Balancing test: taking {min_test_size} from each class")
    
    # Combine validation sets (balanced)
    val_combined = []
    for text in news_val[:min_val_size]:
        val_combined.append(f"__label__high {text}\n")
    for text in crawl_val[:min_val_size]:
        val_combined.append(f"__label__low {text}\n")
    
    # Combine test sets (balanced)
    test_combined = []
    for text in news_test[:min_test_size]:
        test_combined.append(f"__label__high {text}\n")
    for text in crawl_test[:min_test_size]:
        test_combined.append(f"__label__low {text}\n")
    
    # Shuffle
    random.shuffle(val_combined)
    random.shuffle(test_combined)
    
    # Write files
    with open(f"{output_dir}/combined_val.txt", 'w', encoding='utf-8') as f:
        f.writelines(val_combined)
    
    with open(f"{output_dir}/combined_test.txt", 'w', encoding='utf-8') as f:
        f.writelines(test_combined)
    
    print(f"Created balanced combined_val.txt with {len(val_combined)} examples ({len(val_combined)//2} each class)")
    print(f"Created balanced combined_test.txt with {len(test_combined)} examples ({len(test_combined)//2} each class)")

def main():
    # Set random seed
    random.seed(42)
    
    # File paths (relative to repository root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    news_file = os.path.join(repo_root, "data/raw/2016_0000.parquet")
    crawl_file = os.path.join(repo_root, "data/raw/common_crawl.json")
    output_dir = os.path.join(repo_root, "data/processed")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(news_file):
        print(f"Error: News file not found: {news_file}")
        return
    
    if not os.path.exists(crawl_file):
        print(f"Error: Crawl file not found: {crawl_file}")
        return
    
    print("Creating datasets...")
    print("=" * 50)
    
    # Sample news data (positive examples)
    news_val, news_test = sample_news_data(news_file, output_dir)
    
    print("\n" + "=" * 50)
    
    # Sample crawl data (negative examples)
    crawl_val, crawl_test = sample_crawl_data(crawl_file, output_dir)
    
    print("\n" + "=" * 50)
    
    # Create combined validation and test files
    create_combined_files(news_val, news_test, crawl_val, crawl_test, output_dir)
    
    print("\n" + "=" * 50)
    print("Dataset creation completed!")
    print(f"Files created in: {output_dir}/")
    print("- train_positive.txt")
    print("- train_negative.txt") 
    print("- combined_val.txt")
    print("- combined_test.txt")
    print("\nNext steps:")
    print(f"python train_fasttext.py --positive {output_dir}/train_positive.txt")
    print(f"python predict_fasttext.py --input {output_dir}/combined_test.txt")

if __name__ == "__main__":
    main()