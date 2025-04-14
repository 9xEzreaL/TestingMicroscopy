# Change Log

### ToDos:
    - Testing code OOR
    - Separate Enc/ Dec
    - zarr
    - neuroglancer

### Issues
    - need to fix exponential normalization (norm_method = exp w/ exp_trd and exp_ftr)
    - sync transfer save image issue

## [0.1.5] -2025-02-06
    - Correct readme

## [0.1.4] - 2025-02-01
    - Remove cpu/gpu async transfer avoid save image issue
    - Add flag --augmentation to choose do TTA while encode or decode (decode will be faster)
    - Add flag --roi to assigned subfolder naming

## [0.1.3] - 2025-01-20
    - Optimize decoder to cpu too slow issue by using async transfer,save up to 50% of time.
    - Fix uint16 brightness issue

## [0.1.2] - 2025-01-14
    - Refactor code OOR principle
    - Can take stacks of 2D tif as input
    - Separate encoder and decoder operation, latent z saved in hbranch folder
    - Support fp16 while inference
    - Optimize assemble images function
## [0.1.1] - 2024-12-12
    - optimize unnecessary "permute" operation while TTA
    - input_augmentation params using flipX/Y instead of flip2/3
    - add --mc args represent monte carlo over N time 
    - save results path assigned by config yaml
## [0.1.0] - 2024-12-10
    - initial release
### Added
### Changed
### Fixed