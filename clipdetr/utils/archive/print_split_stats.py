from collections import Counter
from clipdetr.datasets.yolo_dataset import YOLODataset
from clipdetr.tokenizer import SimpleTokenizer

root='data/weak-2/final_dataset'
train_ds = YOLODataset(root=root, split='stratified_train', tokenizer=SimpleTokenizer())
val_ds = YOLODataset(root=root, split='stratified_val', tokenizer=SimpleTokenizer())
print('train length =', len(train_ds))
print('val length =', len(val_ds))

counter = Counter()
for i in range(len(train_ds)):
    _, _, boxes, class_ids = train_ds[i]
    for c in class_ids.tolist():
        counter[c] += 1

print('\nTop class counts (id:count)')
for cid, cnt in counter.most_common(10):
    print(cid, cnt)
