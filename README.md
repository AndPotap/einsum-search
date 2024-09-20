# Searching for Efficient Linear Layers over a Continuous Space of Structured Matrices
This repository contains the code for the paper [Searching for Efficient Linear Layers over a Continuous Space of Structured Matrices](https://arxiv.org/abs/2406.06248)
by Andres Potapczynski, Shikai Qiu, Marc Finzi, Christopher Ferri, Zixi Chen, Micah Goldblum, Bayan Bruss, Christopher
De Sa and Andrew Gordon Wilson.

<!---
<table>
  <tr>
    <td><img src="./assets/3d.png" alt="Image 1" width="400"/></td>
    <td><img src="./assets/vecs_gpt2_smallvocab.png" alt="Image 2" width="300"/></td>
  </tr>
</table>
-->

<figure>
  <img src="./assets/3d_v3.png" alt="Fig">
</figure>

Dense linear layers are the dominant computational bottleneck in large neural networks, presenting a critical need for more efficient alternatives.
Previous efforts focused on a small number of hand-crafted structured matrices and neglected to investigate whether these structures can surpass dense layers in terms of compute-optimal scaling laws when both the model size and training examples are optimally allocated.
In this work, we present a unifying framework that enables searching among all linear operators expressible via an Einstein summation.
This framework encompasses many previously proposed structures, such as low-rank, Kronecker, Tensor-Train, Block Tensor-Train (BTT), and Monarch, along with many novel structures.
To analyze the framework, we develop a taxonomy of all such operators based on their computational and algebraic properties and show that differences in the compute-optimal scaling laws are mostly governed by a small number of variables that we introduce.
Namely, a small ω (which measures parameter sharing) and large ψ (which measures the rank) reliably led to better scaling laws, while
ν (which measures how much structured an operator is) can be varied with often negligible impact on the scaling laws.
Guided by the insight that full-rank structures that maximize parameters per unit of compute perform the best, we propose BTT-MoE, a novel Mixture-of-Experts (MoE) architecture obtained by sparsifying computation in the BTT structure, providing a substantial compute-efficiency gain over dense layers and standard MoE.

Please cite this work as:
```bibtex
@article{potapczynski2024einsum,
    title={{Compute Better Spent: Replacing Dense Layers with Structured Matrices}},
    author={Andres Potapczynski, Shikai Qiu, Marc Finzi, Christopher Ferri, Zixi Chen, Micah Goldblum, Bayan Bruss, Christopher De Sa and Andrew Gordon Wilson},
    journal={arXiv},
    year={2024}
}
```
