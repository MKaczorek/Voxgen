# Voxgen
Code for the UCL Data Science and Machine Learning thesis "Voxgen: Singing voices separation using deep generative priors".
Authors would like to acknowledge work done by Yu et al. in "Zero-Shot Duet Singing Voices Separation with
Diffusion Models" and Mariani et al in "Multi-Source Diffusion Models for Simultaneous Audio Generation and Separation", whose work immensly helped in this project.

## Setup

Install requirements

```bash
pip install -r requirements.txt
```
Make sure to download pre-trained model checkpoints and config file and put them in the desired directory.

## Evaluation

First, download the [MedleyVox](https://github.com/jeonchangbin49/MedleyVox?tab=readme-ov-file) dataset.
Then, run the following command to evaluate the model on the `duet` subset of the dataset.

```bash
python eval_test.py pretrained_model/config.yaml pretrained_model/last-001.ckpt MedleyVox -T 14 --out output_restart_39_cond --cond --hop-length 32768 --self-cond --sampler restart
```

Some important arguments:

1. `-T`: number of diffusion steps
2. `--cond`: use auto-regressive conditioning on the previously generated segment. Without this flag, the model will generate the full length audio at once
3. `--self-cond`: perform auto-regressive conditioning on the generated audio if use together with `--cond`
4. `--hop-length`: the hop length of the moving window
5. `--window`: the size of the moving window. Default to the same length as training data
6. `--sampler`: your choice of the sampling scheme from the diffusion prior

```

### Checkpoint/Logs

The pre-trained singing voice diffusion model can be downloaded [here](https://drive.google.com/drive/folders/1nAj0JDiG70ddr_7UnhszpIiVCh4SzqgW?usp=sharing).
You can find the training logs and unconditional singing samples generated during training on [wandb](https://api.wandb.ai/links/aimless/fqtcyjke).

