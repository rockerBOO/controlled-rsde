# Controlled Rectified Stochastic Differential Equations

An AI guided implementation of controlled ODE for Semantic Image Inversion and Editing.

Asked a LLM to make an implementation after guiding it's approach. Only an example implementation and may have errors. But gives me something to work with when trying to implement it.

## Install

### pip
```
pip install git+https://github.com/rockerBOO/controlled-rsde
```

### uv

```
uv add git+https://github.com/rockerBOO/controlled-rsde
```

## Usage

```python
from controlled_rsde.euler import invert_and_edit_with_euler
# from controlled_rsde.dpmsolver_plus_plus import invert_and_edit_with_dpm_solver

# original_img: Input image tensor [B, C, H, W]
# model: Pre-trained Flux model
# edit_prompt: Text prompt for desired edit
# gamma: Controller guidance for inversion
# eta: Controller guidance for editing
# steps: Number of integration steps
# eta_schedule: Optional time-varying schedule for eta
edited_image = invert_and_edit_with_euler(original_img, model, edit_prompt, gamma=0.5, eta=0.5, steps=100, eta_schedule=None)
```

[Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations](https://arxiv.org/abs/2410.10792)

```
@misc{rout2024semanticimageinversionediting,
      title={Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations},
      author={Litu Rout and Yujia Chen and Nataniel Ruiz and Constantine Caramanis and Sanjay Shakkottai and Wen-Sheng Chu},
      year={2024},
      eprint={2410.10792},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.10792},
}
```
