# GPT-from-scratch

This project follows [Andrej Karpathy's tutorial](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) on building a GPT model from scratch, with modifications to generate text resembling WhatsApp conversations instead of Shakespearean text.

## Features
- Implements a Transformer-based GPT model using PyTorch.
- Tokenizes and preprocesses a WhatsApp chat dataset.
- Trains the model to generate realistic WhatsApp-style conversations.
- Saves and loads model parameters for further training or inference.

## Setup
### Requirements
- Python 3.8+
- PyTorch
- Google Colab (optional but recommended for training on GPUs)

### Installation
Clone the repository:
```bash
git clone https://github.com/Buenobarbie/GPT-from-scratch.git
cd GPT-from-scratch
```

Install dependencies:

```bash
pip install torch
```

## Usage

### Running in Google Colab

The main script is [`gpt_BPE.ipynb`](https://github.com/Buenobarbie/GPT-from-scratch/blob/master/code/gpt_BPE.ipynb), where you can create, load, and train models using Byte Pair Encoding (BPE) for tokenization. To run it in Colab, simply navigate to the file on GitHub and click the "Open with Colab" icon.

I recommend using the **T4 GPU** in Colab for optimal performance, as it’s available for free within the usage limits.




### Training the Model
1. Prepare the WhatsApp dataset (`input.txt` in `data/` folder).
2. Run the notebook to preprocess, train, and evaluate the model.
3. Save the trained model for future use.

### Generating Text
Modify the `prompt` variable to provide a conversation starter:

```python
prompt = "Bárbara: Você é"
prompt_encoded = encode(prompt)
idx = torch.tensor([prompt_encoded], dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
```

### Saving and Loading the Model
To save the trained model:

```python
SAVE = True
if SAVE:
    torch.save(model.state_dict(), '../models/model.pth')
```

To load an existing model:

```python
LOAD = True
if LOAD:
    model.load_state_dict(torch.load('../models/model.pth'))
```

## Customization
You can modify the hyperparameters, dataset, and preprocessing methods to adapt the model to different text styles or improve performance.

## Acknowledgments
- Inspired by Andrej Karpathy’s "GPT from Scratch" tutorial.
- Uses PyTorch for implementation.
