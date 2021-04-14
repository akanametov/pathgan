PathGAN
======================
A Pytorch implementation of Generative Adversarial Network for Heuristics of Sampling-based Path Planning

[arXiv article](https://arxiv.org/pdf/2012.03490.pdf)

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

`RRT*` algorithm:

<a><img src="assets/gan_rrt.png" align="center" height="200px" width="400px"/></a>

Comparing `RRT*` and `Heuristic RRT*`:

<a><img src="assets/rrt_vs_hrrt.png" align="center" height="200px" width="300px"/></a>

### GAN architecture

Overall GAN architecture:

<a><img src="assets/gan.png" align="center" height="300px" width="400px"/></a>

GAN architecture in details:

<a><img src="assets/detailed_gan.png" align="center" height="270px" width="440px"/></a>

## Dataset


## Training

## Results

## License

This project is licensed under MIT.

## Links

* [Generative Adversarial Network based Heuristics
for Sampling-based Path Planning (arXiv article)](https://arxiv.org/pdf/2012.03490.pdf)

* [GAN Path Finder (arXiv article)](https://arxiv.org/pdf/1908.01499.pdf)
