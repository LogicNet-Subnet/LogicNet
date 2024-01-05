<div align="center">

# Image Generating Subnet <!-- omit in toc -->

---

</div>

## Introduction
Welcome to the Image Generating Subnet project. This README provides an overview of the project's structure and example usage for both validators and miners.

### The Incentivized Internet
- [Discord](https://discord.gg/bittensor)
- [Network](https://taostats.io/)
- [Research](https://bittensor.com/whitepaper)

## Project Structure
- `image_generation_subnet`: Contains base, feature functions, and utilities for validators and miners.
- `neurons`: Contains the validator and miner loop.
- `dependency_modules`: Includes servers for `prompt_generation`, `rewarding`, and `miner_endpoint`.

## Installation
1. Clone the repository.
```bash
git clone https://github.com/NicheTensor/NicheImage.git
```
2. Install the dependencies.
```bash
pip install -r requirements.txt
```
3. Install the project.
```bash
pip install -e .
```

## Example Usage
Before running the following commands, make sure to replace the placeholder arguments with appropriate values.

## Start Miner
Before running the following commands, make sure to replace the placeholder arguments with appropriate values.

First you need to start an image generation API on a gpu server that your miners can use. A RTX 3090 GPU is enough for several miners.
```bash
python dependency_modules/miner_endpoint/app.py --port <port> --model_name <model_name>
```

Then you can run several miners using the image generation API:
```bash
pm2 start python --name "miner" -- -m neurons.miner.miner \
--netuid <netuid> \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--generate_endpoint <your_miner_endpoint>/generate \
--info_endpoint <your_miner_endpoint>/info \
--axon.port <your_public_port> \
```

**View logs** 
```bash
pm2 logs miner
```

## Start Validator

Requirements: A validator only needs a cpu server to validate by using our free to use APIs for checking image generation. This is the default setting and requires no configuration.

However, it is possible to run your own image checking APIs if you prefer. This does require a GPU with min 20 GB of ram. You can see how to do this [here.](./dependency_modules/README.md)

### Start Validator

```bash
pm2 start python --name "validator" -- -m neurons.validator.validator \
--netuid <netuid> \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--axon.port <your_public_port> \
```

**View logs** 
```bash
pm2 logs validator
```

