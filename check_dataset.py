#!/usr/bin/env python3
"""Quick dataset statistics check"""
from pathlib import Path

train_labels = Path('../datasets/Aero_dataset/train/labels')
label_files = list(train_labels.glob('*.txt'))

class_counts = {}
total_boxes = 0
empty_files = 0

for lf in label_files:
    with open(lf, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            empty_files += 1
            continue
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                class_counts[cls] = class_counts.get(cls, 0) + 1
                total_boxes += 1

print(f'Total label files: {len(label_files)}')
print(f'Empty files: {empty_files}')
print(f'Total boxes: {total_boxes}')
print(f'Class distribution: {dict(sorted(class_counts.items()))}')

# Check valid split
valid_labels = Path('../datasets/Aero_dataset/valid/labels')
valid_files = list(valid_labels.glob('*.txt'))
print(f'\nValid split: {len(valid_files)} files')

valid_counts = {}
for lf in valid_files:
    with open(lf, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                valid_counts[cls] = valid_counts.get(cls, 0) + 1

print(f'Valid class distribution: {dict(sorted(valid_counts.items()))}')

