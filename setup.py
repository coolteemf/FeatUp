# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['featup',
 'featup.adaptive_conv_cuda',
 'featup.datasets',
 'featup.featurizers',
 'featup.featurizers.dinov2',
 'featup.featurizers.dinov2.layers',
 'featup.featurizers.modules']

package_data = \
{'': ['*'], 'featup': ['configs/*']}

install_requires = \
['ipykernel>=6.29.3,<7.0.0',
 'kornia>=0.7.2,<0.8.0',
 'matplotlib>=3.8.3,<4.0.0',
 'numpy>=1.26.4,<2.0.0',
 'omegaconf>=2.3.0,<3.0.0',
 'pytorch-lightning>=2.2.1,<3.0.0',
 'scikit-learn>=1.4.1.post1,<2.0.0',
 'timm==0.4.12',
 'torch>=2.2.1,<3.0.0',
 'torchmetrics>=1.3.2,<2.0.0',
 'torchvision>=0.17.1,<0.18.0',
 'tqdm>=4.66.2,<5.0.0']

setup_kwargs = {
    'name': 'featup',
    'version': '0.1.0',
    'description': '',
    'long_description': '# FeatUp: A Model-Agnostic Framework for Features at Any Resolution\n###  ICLR 2024\n\n\n[![Website](https://img.shields.io/badge/FeatUp-%F0%9F%8C%90Website-purple?style=flat)](https://aka.ms/featup) [![arXiv](https://img.shields.io/badge/arXiv-2403.10516-b31b1b.svg)](https://arxiv.org/abs/2403.10516) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhamilton723/FeatUp/blob/main/example_usage.ipynb)\n[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FeatUp-orange)](https://huggingface.co/spaces/mhamilton723/FeatUp)\n[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/featup-a-model-agnostic-framework-for/feature-upsampling-on-imagenet)](https://paperswithcode.com/sota/feature-upsampling-on-imagenet?p=featup-a-model-agnostic-framework-for)\n\n[Stephanie Fu*](https://stephanie-fu.github.io/),\n[Mark Hamilton*](https://mhamilton.net/),\n[Laura Brandt](https://people.csail.mit.edu/lebrandt/),\n[Axel Feldman](https://feldmann.nyc/),\n[Zhoutong Zhang](https://ztzhang.info/),\n[William T. Freeman](https://billf.mit.edu/about/bio)\n*Equal Contribution.\n\n![FeatUp Overview Graphic](https://mhamilton.net/images/website_hero_small-p-1080.jpg)\n\n*TL;DR*:FeatUp improves the spatial resolution of any model\'s features by 16-32x without changing their semantics.\n\nhttps://github.com/mhamilton723/FeatUp/assets/6456637/8fb5aa7f-4514-4a97-aebf-76065163cdfd\n\n\n## Contents\n<!--ts-->\n   * [Install](#install)\n   * [Using Pretrained Upsamplers](#using-pretrained-upsamplers)\n   * [Fitting an Implicit Upsampler](#fitting-an-implicit-upsampler-to-an-image)\n   * [Coming Soon](coming-soon)\n   * [Citation](#citation)\n   * [Contact](#contact)\n<!--te-->\n\n## Install\n\n### Pip\nFor those just looking to quickly use the FeatUp APIs install via:\n```shell script\npip install git+https://github.com/mhamilton723/FeatUp\n```\n\n### Local Development\nTo install FeatUp for local development and to get access to the sample images install using the following:\n```shell script\ngit clone https://github.com/mhamilton723/FeatUp.git\ncd FeatUp\npip install -e .\n```\n\n## Using Pretrained Upsamplers\n\nTo see examples of pretrained model usage please see our [Collab notebook](https://colab.research.google.com/github/mhamilton723/FeatUp/blob/main/example_usage.ipynb). We currently supply the following pretrained versions of FeatUp\'s JBU upsampler:\n\n| Model Name | Checkpoint                                                                                                                         | Torch Hub Repository | Torch Hub Name |\n|------------|------------------------------------------------------------------------------------------------------------------------------------|----------------------|----------------|\n| DINO       | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dino16_jbu_stack_cocostuff.ckpt)   | `mhamilton723/FeatUp`  | `dino16`        |\n| DINO v2    | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dinov2_jbu_stack_cocostuff.ckpt)   | `mhamilton723/FeatUp`  | `dinov2`         |\n| CLIP       | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/clip_jbu_stack_cocostuff.ckpt)     | `mhamilton723/FeatUp`  | `clip`           |\n| ViT        | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/vit_jbu_stack_cocostuff.ckpt)      | `mhamilton723/FeatUp`  | `vit`            |\n| ResNet50   | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/resnet50_jbu_stack_cocostuff.ckpt) | `mhamilton723/FeatUp`  | `resnet50`       |\n\nFor example, to load the FeatUp JBU upsampler for the DINO backbone:\n\n```python\nupsampler = torch.hub.load("mhamilton723/FeatUp", \'dino16\')\n```\n\n## Fitting an Implicit Upsampler to an Image\n\nTo train an implicit upsampler for a given image and backbone first clone the repository and install it for \n[local development](#local-development). Then run\n\n```python\ncd featup\npython train_implicit_upsampler.py\n```\n\nParameters for this training operation can be found in the [implicit_upsampler config file](featup/configs/implicit_upsampler.yaml).\n\n\n\n\n## Coming Soon:\n\n- Training your own FeatUp joint bilateral upsampler\n- Simple API for Implicit FeatUp training\n- Pretrained JBU models without layer-norms \n\n\n## Citation\n\n```\n@inproceedings{\n    fu2024featup,\n    title={FeatUp: A Model-Agnostic Framework for Features at Any Resolution},\n    author={Stephanie Fu and Mark Hamilton and Laura E. Brandt and Axel Feldmann and Zhoutong Zhang and William T. Freeman},\n    booktitle={The Twelfth International Conference on Learning Representations},\n    year={2024},\n    url={https://openreview.net/forum?id=GkJiNn2QDF}\n}\n```\n\n## Contact\n\nFor feedback, questions, or press inquiries please contact [Stephanie Fu](mailto:fus@mit.edu) and [Mark Hamilton](mailto:markth@mit.edu)\n',
    'author': 'Your Name',
    'author_email': 'you@example.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
