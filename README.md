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

```bash
conda env create -n ProRiboGen -f ProRiboGen.yml


## Training

## Sampling

## Motif extraction

