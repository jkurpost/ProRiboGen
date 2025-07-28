# ProRiboGen

ProRiboGen is a diffusion language model-based general generative model, designed to generate RNA sequences that conform to any RNA-binding proteins (RBPs) binding preferences. ProRiboGen enables de novo inference of RBPs' binding preferences and can be adopted for motif discovery and RBP binding site identification.

## Installation

### Environment Requirements

- **Python**: 3.11.10  
- **CUDA Support**: CUDA 12.4 (via `pytorch-cuda`)

### Key Dependencies

  - `pytorch==2.5.1`
  - `torchvision==0.20.1`
  - `torchaudio==2.5.1`
  - `transformers==4.46.3`
  - `tokenizers==0.20.3`
  - `sentencepiece==0.2.0`
  - `safetensors==0.4.5`
  - `huggingface-hub==0.26.3`

### Create Conda Environment

```
conda env create -n ProRiboGen -f ProRiboGen.yml
```

## Training

Train a new ProRiboGen by using the command below
```
python trainer_v2.2.py --conf /path/to/training_config_file
```
The training config includes following needs to be specific by users:

"dir_protein_features": which is the dir to the protein feature files, save as .npz format
"file_p_r", the path to the file with protein ids and RNA ids
"file_rna_tokens" the path to RNA tokens
"active_checkpoint" the checkpoint model
"dir_out" the trained model will be saved

Training is supported with DDP 
## Sampling

## Motif extraction

