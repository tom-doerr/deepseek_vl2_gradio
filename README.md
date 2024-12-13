# DeepSeek-VL2 Gradio Demo

A Gradio web interface for the DeepSeek-VL2 visual language model that enables interactive visual question-answering and image understanding.

## Features

- Support for multiple DeepSeek-VL2 model sizes (tiny, small, base)
- Multiple image upload capability
- Interactive text prompting
- Command-line option to fix model size
- Easy deployment with port forwarding

## Requirements

- Python 3.10+
- CUDA-capable GPU
- Required packages:
  - torch
  - gradio
  - transformers
  - deepseek-vl
  - Pillow

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install torch gradio transformers deepseek-vl Pillow
```

## Usage

### Local Development

Run the app with default settings (interactive model selection):
```bash
python app.py
```

Run with a specific model size:
```bash
python app.py --model tiny  # or small, base
```

### Remote Deployment

To sync files and set up port forwarding to a remote server:
```bash
./sync.sh
```

This will:
1. Sync the current directory to the remote server
2. Set up port forwarding from remote port 7860 to local port 7860
3. Allow access to the Gradio interface at http://localhost:7860

To stop port forwarding:
```bash
kill $(cat .port_forward.pid) && rm .port_forward.pid
```

## Model Sizes

- **tiny**: Fastest, lowest resource usage
- **small**: Balanced performance and resource usage
- **base**: Best performance, highest resource usage

## Example Prompts

- "Describe what you see in this image."
- "What objects are present in this scene?"
- "What is the main activity happening in this image?"
- "Can you describe the colors and composition of this image?"

## License

[Add your license information here]
