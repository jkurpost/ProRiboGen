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
Config file example is provided at `config/training_config.json`

The following parameters need to be specified by the user for the training process:

- **`dir_protein_features`**: Directory containing the protein feature files, saved in `.npz` format.
  
- **`file_p_r`**: Path to the file containing the mapping between protein IDs and RNA IDs.

- **`file_rna_tokens`**: Path to the file with RNA tokens.

- **`active_checkpoint`**: Path to the pre-trained checkpoint model (if resuming training or fine-tuning).

- **`dir_out`**: Directory where the trained model will be saved.

- **`num_epochs`** Epochs for training

Ensure that these paths are correctly configured in your training setup before starting the process.

Training is supported with DDP 
## Sampling

Generate RNAs for a target RBPs using the command below
```
python sampler.py --conf /path/to/sampling_config_file
```

Config file example is provided at `config/sampling_config.json`

The following parameters need to be specified by the user for the training process:

- **`dir_protein_features`**: Directory containing the protein feature files, saved in `.npz` format.
  
- **`file_p`**: Path to the protein ids for generation.

- **`dir_out`**: Directory where the generated result will be saved.

- **`num_per_protein`**: Generated RNA count for a specific RBP.

- **`dit_pth`**: Path to the trained ProRiboGen model.

## Motif extraction

