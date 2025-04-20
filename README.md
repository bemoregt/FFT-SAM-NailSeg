# FFT-SAM-NailSeg

Fast Nail Segmentation using Segment Anything Model (SAM) with FFT-based Self-Attention

## Overview

This project implements a modified version of the Segment Anything Model (SAM) for nail segmentation tasks. The standard self-attention mechanism in the transformer blocks is replaced with an FFT-based self-attention to significantly improve computational efficiency.

The key innovation is the use of Fast Fourier Transform (FFT) to reduce the complexity of self-attention from O(n²) to O(n log n), enabling faster training and inference while maintaining segmentation quality.

## Features

- **FFT-based Self-Attention**: Replaces standard self-attention with a more efficient FFT-based implementation
- **Fine-tuning of SAM**: Adapts pre-trained SAM models for nail segmentation
- **Custom Segmentation Head**: Simple segmentation head for binary classification (nail vs. background)
- **Nail Segmentation Dataset Support**: Custom dataset loader for nail segmentation datasets
- **MPS Support**: Works on Apple Silicon (M1/M2) GPUs

## How It Works

### FFT Self-Attention

The conventional self-attention mechanism computes attention scores between all pairs of tokens, resulting in quadratic computational complexity. Our FFT-based approach:

1. Projects queries, keys, and values as in standard attention
2. Applies FFT to queries and keys
3. Performs element-wise multiplication in the frequency domain (equivalent to convolution in time domain)
4. Applies inverse FFT to return to the time domain
5. Combines the result with values

This reduces complexity from O(n²) to O(n log n), making it much more efficient for high-resolution images.

## Implementation Details

The code includes several key components:

1. **FFTSelfAttention**: Custom implementation of self-attention using FFT
2. **ModifiedSAM**: Adaptation of SAM for binary segmentation with a lightweight segmentation head
3. **NailSegmentationDataset**: Custom dataset class for loading nail images and corresponding masks
4. **Training and Evaluation**: Functions to train and evaluate the model with Dice coefficient metrics

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/bemoregt/FFT-SAM-NailSeg.git
cd FFT-SAM-NailSeg

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python train_sam_finetune.py
```

You may need to adjust the path to your nail segmentation dataset in the code:

```python
# In train_sam_finetune.py
data_dir = "/path/to/your/nail_seg_dataset"
```

The expected dataset structure is:
```
nail_seg/
└── trainset_nails_segmentation/
    ├── image1.jpg
    ├── image2.jpg
    ├── ...
    └── labels/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Results

The FFT-based attention mechanism achieves comparable segmentation quality to standard self-attention while significantly reducing training time and memory usage. This makes it particularly suitable for deployment on resource-constrained devices.

## Citation

If you use this code for your research, please cite:

```
@software{FFT-SAM-NailSeg,
  author = {bemoregt},
  title = {FFT-SAM-NailSeg: Fast Nail Segmentation using SAM with FFT Self-Attention},
  year = {2025},
  url = {https://github.com/bemoregt/FFT-SAM-NailSeg}
}
```

## License

MIT

## Acknowledgements

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) from Meta AI Research
- FFT implementation inspired by recent advances in efficient attention mechanisms