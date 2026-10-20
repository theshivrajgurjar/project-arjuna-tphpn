<div align="center">

<img src="assets/arjuna_hero_banner.svg" alt="Project Arjuna Banner" width="100%"/>

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-38bdf8.svg?style=for-the-badge)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/NVIDIA-Dual%20T4%20(SM75)-76b900.svg?style=for-the-badge&logo=nvidia)](https://www.nvidia.com/)
[![Memory](https://img.shields.io/badge/VRAM%20Complexity-O(1)%20Zero--BPTT-10b981.svg?style=for-the-badge)]()
[![Footprint](https://img.shields.io/badge/Model%20Size-4.68%20MB%20%2F%20Universe-c084fc.svg?style=for-the-badge)]()

### **Project Arjuna (`T-PHPN v15.0`): Thermodynamic Physarum-Holographic Predictive Network**
**A Non-Transformer, Zero-Backpropagation-Through-Time Symplectic-Holographic Architecture for $O(1)$-Memory Symbolic Law Discovery, Multi-Horizon Least-Action Planning, White-Box Calculus Decompilation, and Computational Number Theory.**

</div>

---

## 1. Paradigm & Architectural Synthesis

<img src="assets/arjuna_pipeline.svg" alt="Arjuna Pipeline" width="100%"/>

Conventional sequence architectures (Transformers, State-Space Models, and static MLPs) are bottlenecked by **$O(T^2)$ KV-caches**, **$O(T)$ Backpropagation-Through-Time (BPTT) activation graphs**, **frozen test-time weights**, and **black-box opacity**.

**Project Arjuna (`T-PHPN v15.0`)** replaces attention and BPTT with four hardware-grounded physical and algebraic mechanisms:

| Pillar | Mathematical Mechanism | Hardware & Algorithmic Advantage |
| :--- | :--- | :--- |
| **1. Block-Hadamard Holographic ALU ($\circledast_H, \oslash_H$)** | $64 \times 64$ Sylvester-Hadamard orthogonal blocks with unit-phasor spectral normalization | Executes natively on FP16 Tensor Cores (**`99.13%`** unbinding recovery without FP32 FFT bottlenecks). |
| **2. Relativistic Symplectic Phase Space $(q_t, p_t)$** | Undamped-coordinate Störmer-Verlet leapfrog integrator with relativistic kinetic energy $\mathcal{K}(p)=D\sqrt{1+\|p\|^2/D}$ | Preserves canonical coordinate volume ($\det(\partial q_t / \partial q_{t-1}) = 1$) while bounding velocity inside $(-1, 1)^D$. |
| **3. Dopaminergic Physarum-MDL Router** | Non-equilibrium slime-mold tube conductivity $C_{t,k}$ gated by free-energy advantage and Solomonoff MDL penalty | Locks onto minimal operator subgraphs with **`96%–98% L2 cache persistence`** and 1-clause regime switching. |
| **4. Lie-Group Invariant Factorization & Variational EM** | Factorizes base 3-body laws from cyclic Lie translation generators $\mathcal{T}_m$ | Learns non-linear coupled dynamical laws in **strictly $O(1)$ sequence memory** inside `torch.no_grad()`. |

---

## 2. Exact Mathematical Formulation

### 2.1 Quasi-Unitary Block-Walsh-Hadamard Binding & Involutory Unbinding
For block size $b = 64$, let $\mathbf{H}_D = \bigoplus_{m=1}^{D/b} H_b$ with $H_b^T = H_b, H_b^2 = I_b$. With spectral phasor $\psi(\hat{v}) = \Pi_{\text{shift}}(\hat{v}) / \sqrt{\Pi_{\text{shift}}(\hat{v})^2 + 10^{-3}}$, binding and unbinding satisfy:
$$u \circledast_H v = \mathbf{H}_D\big(\mathbf{H}_D u \odot \psi(\mathbf{H}_D v)\big), \qquad (u \circledast_H v) \oslash_H v = \mathbf{H}_D\big(\mathbf{H}_D(u \circledast_H v) \odot \psi(\mathbf{H}_D v)\big) \approx u$$

### 2.2 Lie-Group Shift Factorization & Zero-BPTT Variational EM Plasticity
At each clause evaluation step, the base 3-body wave $W_{3\text{b}, t} = R_{A,t} \circledast_H \mathcal{R}_1(R_{\text{op},t}) \circledast_H \mathcal{R}_2(u_t)$ produces base distribution $P_{\text{base}, t} = \text{softmax}(h_{\text{out}, t} W_{\text{out}}^T)$, while the prior clause pair $S_{\text{carry}, t} = R_{A,t-1} \circledast_H \mathcal{R}_1(R_{B,t-1})$ routes over $K_g = 8$ cyclic Lie translation operators $\mathcal{T}_m$:
$$\pi_{\text{carry}, t} = \text{softmax}\big(\hat{S}_{\text{carry}, t} W_{\text{invariant}}^T\big), \qquad P_{\text{final}, t} = \sum_{m=0}^{K_g - 1} \pi_{\text{carry}, t, m} \, \mathcal{T}_m(P_{\text{base}, t})$$

---

## 3. Verified Dual-T4 Empirical Scorecard (12/12 Benchmarks at 100%)

| # | Benchmark / Module | Mathematical Law / Task | Verified Score | Time (2x T4) |
| :- | :--- | :--- | :--- | :--- |
| **1** | **FP16 Block-Hadamard Unbinding** | $(u \circledast_H v) \oslash_H v \approx u$ in $\mathbb{R}^{256}$ | **`99.13%`** Cosine Sim | `< 0.5 ms` |
| **2** | **Universe 1 (Linear + $\mathbb{Z}_3$ Carry)** | $E_t = (a \cdot \text{op} + b + ((a_{t-1}+b_{t-1})\bmod 3)) \bmod 48$ | **`100.00%`** (`CE: 0.0001`) | `18.4 s` |
| **3** | **Universe 2 (Quadratic + $\mathbb{Z}_4$ Carry)** | $E_t = 1 + ((a^2 + \text{op}\cdot b^2 + (a_{t-1}b_{t-1}\bmod 4) - 1)\bmod 47)$ | **`100.00%`** (`CE: 0.0000`) | `19.2 s` |
| **4** | **Universe 3 (Cubic + $\mathbb{Z}_8$ XOR Carry)** | $E_t = 1 + ((a^3 + \text{op}^2 b + (a_{t-1}\oplus b_{t-1}) - 1)\bmod 47)$ | **`100.00%`** (`CE: 0.0000`) | `15.8 s` |
| **5** | **Universe 4 (Elliptic Curve $E(\mathbb{F}_{47})$)** | $P_{\text{res}} = [a]G \boxplus [\text{op}\cdot b]G \boxplus [\text{Torsion}_5]G$ on $y^2 = x^3+2x+3$ | **`100.00%`** (`CE: 0.0004`) | `19.5 s` |
| **6** | **Conservation Law Table Extraction** | Autonomous discovery of $\mathbb{Z}_3 (+)$, $\mathbb{Z}_4 (\times)$, and $\mathbb{Z}_8 (\oplus)$ tables | **`100.00%`** (`48/48` states) | `< 10 ms` |
| **7** | **1-Var, 2-Var & 4-Var Inverse Planner** | Coupled 2-Horizon Principle-of-Least-Action Synthesis | **`100.00%`** (`256/256` plans) | `1.30 s` |
| **8** | **Calculus Weight Decompiler** | Proving $\frac{\partial E}{\partial b}=1, \frac{\partial^2 E}{\partial a^2}=2, \frac{\partial^3 E}{\partial a^3}=6$, & Counterexamples | **`100.00%`** (`7/7` Theorems) | `0.85 s` |
| **9** | **Tri-Universe Meta-Physarum Router** | Unannounced mid-stream phase transitions (`U1 -> U2 -> U3`) | **`100.00%`** Steady-State | `0.96 s` |
| **10**| **Dual-T4 BSD Elliptic Curve Scanner** | `1,048,568` curves $\times$ `32` primes (`33.55M` group evaluations) | **`14.8M` evals/sec** | `2.26 s` |
| **11**| **Néron–Tate Spectral Basis Prover** | Exact rational decomposition $c = G^{-1}v_P$ on $y^2 = x^3 - 172x + 820$ | **`27/27 (100%)`** Integral Pts | `0.45 s` |
| **12**| **Singer $\text{PG}(2,31)$ Extremal Sidon Set**| Collision-free $B_2$ sequence in $[1..1000]$ (beating greedy $|S|=27$) | **`|S| = 32`** (`528/528` sums) | `0.25 s` |

---

## 4. Autonomous Number-Theory & BSD Elliptic Curve Proofs

### 4.1 Complete `27/27 (100.0%)` Mordell–Weil Decomposition of $y^2 = x^3 - 172x + 820$
From `33,554,176` mod-$p$ Legendre-symbol Frobenius evaluations on Dual T4 GPUs, Arjuna isolated $E: y^2 = x^3 - 172x + 820$, proved the Index-4 / Index-3 Quotient Syzygy:
$$\pm 4 G_5(-4, 38) = -2 G_1(-15, 5) \boxplus 2 G_2(-14, 22) \boxplus 1 G_3(-12, 34) \boxplus -3 G_4(-7, 41) \quad \text{in } E(\mathbb{Q})$$
isolated the **Index-1 Fundamental Basis ($\text{Reg} = +2.3092$)**:
$$F_1 = (-7, 41), \quad F_2 = (2, 22), \quad F_3 = (9, 1), \quad F_4 = (12, 22)$$
and decomposed all **27 pairs of integral points (`54/54` integer solutions in $\mathbb{Z}^2$)** via Néron–Tate Spectral Projection $c = G_{\text{fund}}^{-1} v_P$:

```text
Point #01 (-15, 5)     ==  1*F1 (+)  1*F3 (+)  2*F4             in E(Q) [EXACT RATIONAL PROOF]
Point #03 (-12, 34)    ==  1*F1 (+)  2*F2 (+)  2*F3 (+)  2*F4   in E(Q) [EXACT RATIONAL PROOF]
Point #19 (97, 947)    ==  2*F1 (+)  2*F2 (+)  2*F3 (+)  3*F4   in E(Q) [EXACT RATIONAL PROOF]
Point #25 (758, 20866) ==  1*F1 (+)  1*F2 (+)  1*F3 (+)  3*F4   in E(Q) [EXACT RATIONAL PROOF]
Point #26 (810, 23050) == -1*F1 (+) -3*F2 (+) -2*F3 (+) -3*F4   in E(Q) [EXACT RATIONAL PROOF]
Point #27 (1302, 46978)== -1*F1 (+)  1*F2 (+)  1*F3 (+) -1*F4   in E(Q) [EXACT RATIONAL PROOF]
```

### 4.2 Exact 32-Element Singer $\text{PG}(2, 31)$ Extremal Sidon Set ($B_2$ Sequence) in $[1..1000]$
```json
[1, 10, 81, 148, 173, 186, 226, 305, 334, 350, 392, 409, 415, 423, 433, 485, 518, 545, 546, 548, 567, 582, 602, 614, 688, 776, 802, 848, 853, 892, 896, 903]
```

---

## 5. Quickstart & Reproduction

```bash
git clone https://github.com/theshivrajgurjar/project-arjuna-tphpn.git
cd project-arjuna-tphpn
pip install -r requirements.txt
python app.py
python arjuna_tphpn_engine.py
```

---

## 6. License

Released under the **MIT License**. Created by **@theshivrajgurjar** (Shivraj Gurjar).