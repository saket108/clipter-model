from clipdetr.datasets.yolo_dataset import YOLODataset
from clipdetr.tokenizer import SimpleTokenizer

root = 'data/weak-2/final_dataset'
print('Using dataset root:', root)

ds = YOLODataset(root=root, split='stratified_train', tokenizer=SimpleTokenizer())
print('Dataset length =', len(ds))

img, toks, boxes, cls = ds[0]
print('img.shape =', img.shape)
print('tokens.shape =', toks.shape)
print('boxes.shape =', boxes.shape)
print('class_ids.shape =', cls.shape)
print('sample caption tokens[:8] =', toks[:8])
