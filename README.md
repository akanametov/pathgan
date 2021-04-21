PathGAN
======================
A Pytorch implementation of **Generative Adversarial Network for Heuristics of Sampling-based Path Planning**

[Original arXiv paper](https://arxiv.org/pdf/2012.03490.pdf)

[Dataset](https://disk.yandex.ru/d/mgf5wtQrld0ygQ)

## Table of content

- [Structure](#structure)
  - [Searching algorithm](#searching-algorithm)
  - [GAN architecture](#gan-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [License](#license)
- [Links](#links)


## Structure

The overall structure of the PathGAN consists of two things:
1) RRT* searching algorithm and
2) Generative Aversarial Network for promising region generation 

### Searching algorithm

**`RRT*` algorithm:**

<a><img src="assets/gan_rrt.png" align="center" height="150px" width="350px"/></a>

**Comparing `RRT*` and `Heuristic RRT*`:**

<a><img src="assets/rrt_vs_hrrt.png" align="center" height="350px" width="340px"/></a>

### GAN architecture

**Overall `GAN` architecture:**

<a><img src="assets/gan.jpg" align="center" height="400px" width="600px"/></a>

**`GAN` architecture in details:**

<a><img src="assets/detailed_gan.jpg" align="center" height="400px" width="800px"/></a>

## Dataset

[Dataset](https://disk.yandex.ru/d/mgf5wtQrld0ygQ)

## Training

## Results

<a><img src="assets/result.png" align="center" height="400px" width="500px"/></a>

## License

This project is licensed under MIT.

## Links

* [Generative Adversarial Network based Heuristics for Sampling-based Path Planning (arXiv article)](https://arxiv.org/pdf/2012.03490.pdf)

* [Dataset](https://disk.yandex.ru/d/mgf5wtQrld0ygQ)
