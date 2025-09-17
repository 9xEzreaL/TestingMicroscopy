# Microscopy Image Enhancement Inference

This repository contains inference code for microscopy image enhancement using deep learning models. The system supports both encoding and decoding operations with multi-GPU acceleration and various data formats.

## Features

- **Multi-GPU Support**: Efficient parallel processing across multiple GPUs
- **Flexible Input Formats**: Support for 2D/3D TIFF and 2D/3D Zarr formats
- **Large Data Handling**: 3D Zarr format supports scaling up to 100GB+ datasets
- **Multiple Output Formats**: Save results as TIFF, Zarr, or skip saving
- **Data Type Options**: Output in float32, uint16, or uint8
- **Configuration-Driven**: All settings managed through YAML configuration files

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── test_only.py          # Encoding inference code
├── test_assemble.py      # Decoding inference code  
├── test/                 # Configuration files
│   ├── config.yaml       # Main configuration
│   ├── config_example.yaml
│   └── ...
├── utils/                # Utility modules
├── networks/             # Model architectures
├── models/               # Model definitions
└── sample/               # Example outputs
```

## Configuration

The system uses YAML configuration files located in the `test/` directory. Key configuration sections:

### Basic Settings
```yaml
SOURCE: '/path/to/model/logs'
DESTINATION: '/path/to/output'
N_resolution: 8  # Super-resolution factor
N_GPUS: 2        # Number of GPUs for decoding
```

### Model Configuration
```yaml
DPM:  # or VMAT
  dataset: "DPM4X"
  prj: "DPM4X/ae/cut/1/"
  epoch: 800
  model_type: AE
  hbranchz: true
  image_path: ["/ori/3-2ROI000.tif", "/ft0/3-2ROI000.tif"]
  hbranch_path: "/path/to/hbranch"
```

### Processing Parameters
```yaml
assemble_params:
  C: [32, 32, 32]     # Cropped pixels
  S: [64, 64, 64]     # Overlapping pixels
  dx_shape: [32, 256, 256]  # Inference patch size
  zrange: [0, 449]    # Z coordinate range
  xrange: [0, 769]    # X coordinate range
  yrange: [0, 769]    # Y coordinate range
```

## Usage

### Encoding (test_only.py)

For encoding operations that convert input images to latent representations:

```bash
python test_only.py --gpu --config config_name --augmentation decode --fp16 --option DPM --testcube
```

**Key Parameters:**
- `--gpu`: Enable GPU acceleration
- `--config`: Configuration file name (without .yaml extension)
- `--augmentation`: Augmentation strategy (decode recommended for speed)
- `--fp16`: Use half-precision for faster inference
- `--option`: Model variant (DPM or VMAT)
- `--testcube`: Enable full volume processing

### Decoding (test_assemble.py)

For decoding operations that reconstruct images from latent representations:

```bash
python test_assemble.py --gpu --config config_name --augmentation decode --fp16 --option DPM --reslice --testcube
```

**Key Parameters:**
- `--testcube`: Run decode or not
- `--gpu`: Enable GPU acceleration
- `--config`: Configuration file name
- `--augmentation`: Augmentation strategy
- `--fp16`: Use half-precision
- `--assemble_method`: save as tiff zarr or none
- `--option`: Model variant
- `--reslice`: Reslice original images or not

## Supported Data Formats

### Input Formats
- **2D TIFF**: Individual 2D images
- **3D TIFF**: 3D image stacks
- **2D Zarr**: 2D Zarr arrays
- **3D Zarr**: 3D Zarr arrays (supports 100GB+ datasets)

### Output Formats
- **TIFF**: Standard TIFF format
- **Zarr**: Compressed Zarr format with Blosc compression
- **No Save**: Process without saving (for testing)

### Data Types
- **float32**: Full precision (default)
- **uint16**: 16-bit unsigned integer
- **uint8**: 8-bit unsigned integer

## Performance Benchmarks (100GB Dataset)

### Encoding Performance (A6000 Single GPU, Latent Stored as Zarr)

| Input Patch Size | Latent Size | Total Time | Time per Patch |
|:----------------:|:-----------:|:----------:|:--------------:|
| 32×256×256 | 32×4×32×32 | 4238s |    0.1948s     |
| 48×384×384 | 48×4×48×48 | 2872s |    0.5249s     |

### Decoding Performance (A6000 GPU, Output Saved as TIFF)

| Input Latent Size | Output Size | GPUs | Total Time | Time per Patch |
|:-----------------:|:-----------:|:----:|:----------:|:--------------:|
| 32×4×32×32 | 256³ | 1 | 15241s | 0.7007s |
| 48×4×48×48 | 384³ | 1 | 13457s | 2.4594s |
| 48×4×48×48 | 384³ | 2 | 8419s | 1.5386s |
| 48×4×48×48 | 384³ | 4 | 6039s | 1.1038s |

**Note**: Saving as Zarr format will be slower than TIFF due to compression overhead.

## Multi-GPU Processing

The decoding pipeline (`test_assemble.py`) supports multi-GPU processing:

1. **Data Distribution**: Input data is distributed across available GPUs
2. **Parallel Processing**: Each GPU processes different patches simultaneously
3. **Result Assembly**: Results are collected and assembled into final output
4. **Memory Efficient**: Uses queue-based processing to manage memory

## Configuration Examples

### Basic Configuration
```yaml
DEFAULT:
  SOURCE: '/path/to/models'
  DESTINATION: '/path/to/output'
  N_resolution: 8
  N_GPUS: 2
  
  upsample_params:
    size: [32, 256, 256]
    
  patch_range:
    d0: [189, 120, 400]
    dx: [32, 256, 256]
    
  norm_method: ["exp", "11"]
  trd: [[100, 424], [0, 4]]
```

### Model-Specific Configuration
```yaml
DPM:
  dataset: "DPM4X"
  prj: "DPM4X/ae/cut/1/"
  epoch: 800
  model_type: AE
  hbranchz: true
  image_path: ["/ori/3-2ROI000.tif", "/ft0/3-2ROI000.tif"]
  hbranch_path: "/path/to/hbranch"
```

## Advanced Features

### Data Normalization
- **Exponential normalization**: For fluorescence data
- **Standard normalization**: Min-max scaling
- **Custom thresholds**: Configurable normalization parameters

### Augmentation Strategies
- **None**: No augmentation
- **Transpose**: Spatial transposition
- **FlipX/FlipY**: Horizontal/vertical flipping
- **Combined**: Multiple augmentations for ensemble results

### Memory Management
- **Chunked Processing**: Large datasets processed in chunks
- **Queue-based I/O**: Asynchronous file writing
- **GPU Memory Optimization**: Efficient memory usage across GPUs

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use fewer GPUs
2. **File Not Found**: Check configuration paths
3. **Zarr Errors**: Ensure Zarr files are properly formatted

### Performance Tips
1. Use `--fp16` for faster inference
2. Use `--augmentation decode` for better performance
3. Configure appropriate chunk sizes for Zarr files
4. Use multiple GPUs for large datasets

## Example Commands

```bash
# Encode with full precision
python test_only.py --gpu --config config_dpm --save ori recon xy --augmentation decode --option DPM --testcube

# Decode with half precision and multiple GPUs
python test_assemble.py --gpu --config config_dpm --augmentation decode --fp16 --option DPM --reslice
```

## License

This project is for research purposes. Please ensure you have appropriate licenses for any models or datasets used.
