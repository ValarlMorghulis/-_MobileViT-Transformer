# 数学建模大作业

## 1. 论文选择

在众多的学术论文之中，经过仔细地筛选与考量，最终选择了论文[MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](./MobileViT%20Light-weight,%20General-purpose,%20and%20Mobile-friendly%20Vision%20Transformer.pdf)。之所以选择这篇论文，是因为它在视觉 Transformer 领域有着独特的研究视角和重要的研究成果，其探讨的关于轻量化、通用性以及适用于移动设备等相关内容，与当下的技术发展趋势紧密结合，对于学习和理解视觉 Transformer 模型有着重要的意义。

## 2. 论文翻译

按照[数学建模竞赛要求](https://www.overleaf.com/read/pckcchwxhxtd#61256c)，改写翻译论文，最终成果：[MobileViT：轻量化，通用，适用于移动设备的视觉Transformer](./MobileViT_轻量化_通用_适用于移动设备的视觉Transformer.pdf)。在翻译的过程中，我也在文中增添了对该论文提出的方法的一些优缺点以及未来优化方向的思考。

## 3. 代码复现

在论文的实践应用方面，论文的原代码已经在GitHub上公开，项目名为[CVNets: A library for training computer vision networks](https://github.com/apple/ml-cvnets)。这个开源项目为计算机视觉网络的训练提供了一个强大的工具库。为了更好地理解和应用这些代码，我提供了一份详细的[代码链接与使用说明](./代码链接与使用说明.md)，便于快速上手。

同时，我也结合论文中的提到的方法和GitHub上的开源代码，使用PyTorch框架复现了MobileViT模型。这一过程不仅加深了我对论文理论的理解，也锻炼了我的编程实践能力。复现的代码已经整理成文件[MobileViT.py](./MobileViT.py)。
