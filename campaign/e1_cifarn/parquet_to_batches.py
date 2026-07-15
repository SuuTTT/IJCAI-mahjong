#!/usr/bin/env python3
"""Convert HF uoft-cs/cifar10 parquet files into the canonical
cifar-10-batches-py pickle format (data_batch_1..5, test_batch) so the
experiment code can use the standard loader. Order is preserved as-is;
alignment with CIFAR-10N is verified downstream against clean_label."""
import io, os, pickle, sys
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
out = os.path.join(data_dir, 'cifar-10-batches-py')
os.makedirs(out, exist_ok=True)


def decode(parquet_path):
    t = pq.read_table(parquet_path)
    cols = t.column_names
    img_col = 'img' if 'img' in cols else 'image'
    imgs = t.column(img_col).to_pylist()
    labels = t.column('label').to_pylist()
    xs = np.zeros((len(imgs), 3072), dtype=np.uint8)
    for i, rec in enumerate(imgs):
        b = rec['bytes'] if isinstance(rec, dict) else rec
        a = np.asarray(Image.open(io.BytesIO(b)).convert('RGB'), dtype=np.uint8)
        xs[i] = a.transpose(2, 0, 1).reshape(-1)  # HWC -> CHW planes
    return xs, [int(l) for l in labels]


xtr, ytr = decode(os.path.join(data_dir, 'train.parquet'))
xte, yte = decode(os.path.join(data_dir, 'test.parquet'))
assert xtr.shape == (50000, 3072) and xte.shape == (10000, 3072), (xtr.shape, xte.shape)

for i in range(5):
    with open(os.path.join(out, f'data_batch_{i + 1}'), 'wb') as f:
        pickle.dump({b'data': xtr[i * 10000:(i + 1) * 10000],
                     b'labels': ytr[i * 10000:(i + 1) * 10000]}, f)
with open(os.path.join(out, 'test_batch'), 'wb') as f:
    pickle.dump({b'data': xte, b'labels': yte}, f)
print('wrote batches:', sorted(os.listdir(out)))
print('train label head:', ytr[:20])
print('test  label head:', yte[:20])
