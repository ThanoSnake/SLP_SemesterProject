# Physically Interpretable World Models (PIWM)

**Reproducing Four PIWM Principles and Extensions to Dynamics, Uncertainty and Control**

---

## Overview

A world model learns to compress observations into a latent vector and to predict how that latent evolves over time, so control and planning can be done "in the imagination". The problem with standard world models is that the latent space is a black box, no dimension corresponds to a measurable physical quantity, which blocks generalization, safe reuse by a controller, and formal verification.

This project reproduces and systematically implements the four principles for Physically Interpretable World Models (PIWM), originally proposed by Peper et al., on two environments, CartPole and LunarLander. The paper leaves most architectural and training choices unspecified, a large part of the work is covering those ambiguities with well-justified design decisions (two-frame input, split-β KL, residual LSTM, teacher forcing with decaying *p*, curriculum horizon) so that the benefit of the principles becomes visible and measurable.

Beyond the reproduction, we add three extensions :

1. **SINDy** explicit dynamics with : replacing and hybridizing the neural LSTM with sparse, readable difference equations.

2. **Uncertainty** quantification : aleatoric (VAE `logvar`) and epistemic (MC-Dropout) uncertainty directly on the physical dimensions, in real units.

3. **Control** of LunarLander from pixels : the interpretable encoder feeds a classic PID, and the model's "dream" acts as a targeted safety shield.

---

## Setup

```bash
pip install -r requirements.txt
```

Every path is resolved by [`config.py`](config.py) and re-exported to the scripts through `cartpole/paths.py` and `lunarlander/paths.py`. Scripts are run from the repository root and write everything under `OUTPUT_DIR`.

### Data collection

Only needed to regenerate the datasets from scratch.

```bash
python cartpole/dataCollect.py
python lunarlander/dataCollect.py
```

### Baseline

```bash
python cartpole/vae.py
python cartpole/lstm.py       
python cartpole/test_baseline.py
```

### The Four Principles

```bash
python cartpole/vae_p1.py
python cartpole/lstm_p1.py 
python cartpole/test_p1.py
```

```bash
python cartpole/vae_p2.py 
python cartpole/lstm_p2.py 
python cartpole/test_p2.py
```

```bash
python cartpole/vae_p3.py 
python cartpole/lstm_p3.py 
python cartpole/test_p3.py
```

`test_p3.py` compares baseline / semi / weak, so run `vae_p3.py` and `lstm_p3.py` once per
setting — set `SUPERVISION = "semi"` then `"weak"` at the top of both files.

```bash
python cartpole/vae_p4.py 
python cartpole/test_p4.py
```

### Extensions

Uncertainty :

```bash
python cartpole/uncertainty.py
```

Dynamics with SINDy :

```bash
python cartpole/sindy.py                
python cartpole/test_sindy_vs_lstm.py    
python cartpole/fusion_kalman.py         
```

Control :

```bash
python lunarlander/control.py                             
python lunarlander/extension_main_shield_emergency_relaxed.py
```

Both need the encoder/LSTM retrained on the coverage-oriented control dataset
(`LUNARLANDER_CONTROL_VAE` / `LUNARLANDER_CONTROL_LSTM`), which are not in `weights/` — collect
the dataset with `lunar_data_collect_control.py` and retrain with `vae_p1_control.py` and
`lstm_p1_control.py`, or point the two variables at your own checkpoints.

---

## Citations

```bibtex
@inproceedings{peper2025piwm,
  title={Four Principles for Physically Interpretable World Models},
  author={Peper, Jordan and Mao, Zhenjiang and Geng, Yuang and Pan, Siyu and Ruchkin, Ivan},
  booktitle={Proceedings of the 2nd International Conference on Neuro-symbolic Systems (NeuS)},
  series={Proceedings of Machine Learning Research},
  volume={288},
  year={2025}
}
```
