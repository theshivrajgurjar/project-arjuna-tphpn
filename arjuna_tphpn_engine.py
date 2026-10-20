#!/usr/bin/env python3
# ==============================================================================
# PROJECT ARJUNA (T-PHPN v15.0): STANDALONE REFERENCE ENGINE & VERIFICATION SUITE
# ==============================================================================

from fractions import Fraction
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArjunaBlockHadamardALU(nn.Module):
    """Quasi-Unitary 64x64 Block-Walsh-Hadamard Binder & Involutory Unbinder."""

    def __init__(self, dim: int = 256, block_size: int = 64):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.num_blocks = dim // block_size
        h_mat = torch.tensor([[1.0]], dtype=torch.float32)
        for _ in range(int(math.log2(block_size))):
            h_mat = torch.cat(
                [torch.cat([h_mat, h_mat], dim=1), torch.cat([h_mat, -h_mat], dim=1)],
                dim=0,
            )
        self.register_buffer("hadamard_block", h_mat / math.sqrt(float(block_size)))

    @staticmethod
    def algebraic_saturate(x: torch.Tensor) -> torch.Tensor:
        x_s = torch.clamp(x, min=-250.0, max=250.0)
        return (x_s * torch.rsqrt(1.0 + x_s * x_s)).to(dtype=x.dtype)

    @staticmethod
    def algebraic_saturate_deriv(x: torch.Tensor) -> torch.Tensor:
        x_s = torch.clamp(x, min=-250.0, max=250.0)
        inv = torch.rsqrt(1.0 + x_s * x_s)
        return (inv * inv * inv).to(dtype=x.dtype)

    def apply_block_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        xb = x.view(-1, self.num_blocks, self.block_size)
        hb = self.hadamard_block.to(dtype=x.dtype, device=x.device)
        return torch.matmul(xb, hb).view_as(x)

    def _spectral_phasor(self, v: torch.Tensor) -> torch.Tensor:
        hv = self.apply_block_hadamard(v).float()
        hv_s = torch.roll(hv.view(-1, self.num_blocks, self.block_size), 1, dims=-1).view_as(hv)
        hv_c = torch.clamp(hv_s, min=-250.0, max=250.0)
        return (hv_c * torch.rsqrt(hv_c * hv_c + 1e-3)).to(dtype=v.dtype)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.apply_block_hadamard(self.apply_block_hadamard(u) * self._spectral_phasor(v))

    def unbind(self, bound_uv: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.apply_block_hadamard(self.apply_block_hadamard(bound_uv) * self._spectral_phasor(v))


class ArjunaHolographicCartridge(nn.Module):
    """O(sqrt(M_s)) Hierarchical Associative World Memory Bank."""

    def __init__(self, dim: int = 256, num_coarse: int = 32, num_leaf: int = 32, temperature: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_coarse = num_coarse
        self.num_leaf = num_leaf
        self.temperature = temperature
        self.scale = 1.0 / math.sqrt(float(dim))
        self.coarse_keys = nn.Parameter(torch.randn(num_coarse, dim) * self.scale, requires_grad=False)
        self.leaf_keys = nn.Parameter(torch.randn(num_coarse, num_leaf, dim) * self.scale, requires_grad=False)
        self.leaf_values = nn.Parameter(torch.randn(num_coarse, num_leaf, dim) * self.scale, requires_grad=False)

    def retrieve(self, q_mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ck = self.coarse_keys.to(dtype=q_mem.dtype)
        lk = self.leaf_keys.to(dtype=q_mem.dtype)
        lv = self.leaf_values.to(dtype=q_mem.dtype)
        c_star = torch.argmax(torch.matmul(q_mem, ck.t()) * self.scale, dim=-1)
        sk, sv = lk[c_star], lv[c_star]
        f_logits = torch.bmm(sk, q_mem.unsqueeze(-1)).squeeze(-1) * (self.scale * self.temperature)
        s_fine = F.softmax(f_logits.float(), dim=-1).to(dtype=q_mem.dtype)
        return torch.bmm(s_fine.unsqueeze(1), sv).squeeze(1), c_star, s_fine

    def compute_local_plasticity(
        self, q_mem_elig: torch.Tensor, c_star: torch.Tensor, s_fine: torch.Tensor, delta_f: torch.Tensor, oja_rate: float = 0.04
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b_val = torch.bmm(s_fine.unsqueeze(-1).float(), delta_f.unsqueeze(1).float())
        sk = self.leaf_keys[c_star].float()
        rec_q = torch.bmm(s_fine.unsqueeze(1).float(), sk).squeeze(1)
        oja_res = (q_mem_elig.float() - rec_q) * (-oja_rate)
        b_key = torch.bmm(s_fine.unsqueeze(-1).float(), oja_res.unsqueeze(1))

        d_lv = torch.zeros_like(self.leaf_values, dtype=torch.float32)
        d_lk = torch.zeros_like(self.leaf_keys, dtype=torch.float32)
        d_ck = torch.zeros_like(self.coarse_keys, dtype=torch.float32)
        d_lv.index_add_(0, c_star, b_val)
        d_lk.index_add_(0, c_star, b_key)
        d_ck.index_add_(0, c_star, (self.coarse_keys[c_star].float() - q_mem_elig.float()) * oja_rate)
        return d_ck, d_lk, d_lv


class PhysarumMDLRouter(nn.Module):
    """Dopaminergic Slime-Mold Router with Solomonoff MDL Regularization."""

    def __init__(self, dim: int = 256, num_nodes: int = 16, rank: int = 64, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.rank = rank
        self.top_k = top_k
        self.mu = 1.35
        self.rho_c = 0.20
        self.lambda_mdl = 0.10
        scale = 1.0 / math.sqrt(float(dim))
        self.op_u = nn.Parameter(torch.randn(num_nodes, dim, rank) * scale, requires_grad=False)
        self.op_v = nn.Parameter(torch.randn(num_nodes, dim, rank) * scale, requires_grad=False)

    def route_and_step_conductivity(
        self, q: torch.Tensor, conductivity: torch.Tensor, reward_mod: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_all = torch.einsum("bd,ndr->bnr", q, self.op_u.to(dtype=q.dtype))
        p_sq = torch.mean(z_all.float() ** 2, dim=-1)
        mdl_term = torch.log1p(p_sq)
        flux = (conductivity.float() * torch.sqrt(p_sq + 1e-6)) / (1.0 + self.lambda_mdl * mdl_term)
        if reward_mod is not None:
            flux = flux * reward_mod
        f_pow = torch.pow(torch.clamp(flux, min=1e-6), self.mu)
        new_c = (1.0 - self.rho_c) * conductivity.float() + self.rho_c * (f_pow / (torch.mean(f_pow, dim=-1, keepdim=True) + 1e-6))
        new_c = torch.clamp(new_c, min=0.01 / self.num_nodes)
        new_c = (new_c / torch.sum(new_c, dim=-1, keepdim=True)).to(dtype=q.dtype)
        top_f, top_idx = torch.topk(flux, k=self.top_k, dim=-1)
        gates = F.softmax(top_f, dim=-1).to(dtype=q.dtype)
        return new_c, top_idx, gates, flux.to(dtype=q.dtype), torch.mean(mdl_term)

    def compute_hamiltonian_force(self, q: torch.Tensor, top_idx: torch.Tensor, gates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_act = self.op_u.to(dtype=q.dtype)[top_idx]
        v_act = self.op_v.to(dtype=q.dtype)[top_idx]
        z_act = torch.einsum("bd,bkdr->bkr", q, u_act)
        phi_z = ArjunaBlockHadamardALU.algebraic_saturate(z_act)
        f_k = torch.einsum("bkr,bkdr->bkd", phi_z, v_act)
        return torch.sum(f_k * gates.to(dtype=q.dtype).unsqueeze(-1), dim=1), z_act

    def compute_local_operator_plasticity(
        self, q_elig: torch.Tensor, top_idx: torch.Tensor, gates: torch.Tensor, delta_force: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u_act = self.op_u[top_idx].float()
        v_act = self.op_v[top_idx].float()
        qf = q_elig.float()
        z_act = torch.einsum("bd,bkdr->bkr", qf, u_act)
        phi_z = ArjunaBlockHadamardALU.algebraic_saturate(z_act)
        dphi_z = ArjunaBlockHadamardALU.algebraic_saturate_deriv(z_act)
        df_g = delta_force.float().unsqueeze(1) * gates.float().unsqueeze(-1)
        gv = torch.einsum("bkd,bkr->bkdr", df_g, phi_z)
        dz = torch.einsum("bkd,bkdr->bkr", df_g, v_act) * dphi_z
        z_nsq = torch.mean(z_act * z_act, dim=-1, keepdim=True)
        mdl_gz = ((2.0 * self.lambda_mdl * 1e-3 / float(self.rank)) * z_act) / (1.0 + z_nsq)
        gu = torch.einsum("bd,bkr->bkdr", qf, dz + mdl_gz)
        du = torch.zeros_like(self.op_u, dtype=torch.float32)
        dv = torch.zeros_like(self.op_v, dtype=torch.float32)
        flat = top_idx.reshape(-1)
        du.index_add_(0, flat, gu.reshape(-1, self.dim, self.rank))
        dv.index_add_(0, flat, gv.reshape(-1, self.dim, self.rank))
        return du, dv


class SymplecticRelativisticCell(nn.Module):
    """Undamped-Coordinate Relativistic Störmer-Verlet Symplectic Integrator."""

    def __init__(self, dim: int = 256, num_nodes: int = 16, rank: int = 64, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.dt = 0.35
        self.damping = 0.15
        self.confine_omega = 0.25
        self.router = PhysarumMDLRouter(dim, num_nodes, rank, top_k)
        self.rms_scale = nn.Parameter(torch.ones(dim), requires_grad=False)

    @staticmethod
    def relativistic_velocity(p: torch.Tensor) -> torch.Tensor:
        pf = p.float()
        return (pf * torch.rsqrt(1.0 + torch.mean(pf * pf, dim=-1, keepdim=True))).to(dtype=p.dtype)

    def total_conservative_force(self, q: torch.Tensor, top_idx: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        f_p, _ = self.router.compute_hamiltonian_force(q, top_idx, gates)
        f_c = (self.confine_omega * ArjunaBlockHadamardALU.algebraic_saturate(q)).to(dtype=q.dtype)
        return f_p + f_c

    def forward(
        self, q_prev: torch.Tensor, p_prev: torch.Tensor, c_prev: torch.Tensor, f_ext: torch.Tensor, reward_mod: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_t, top_idx, gates, flux, mdl_cost = self.router.route_and_step_conductivity(q_prev, c_prev, reward_mod)
        df = 1.0 - self.damping * self.dt
        hdt = 0.5 * self.dt
        f_prev = self.total_conservative_force(q_prev, top_idx, gates)
        p_half = df * p_prev - hdt * f_prev + hdt * f_ext
        q_t = q_prev + self.dt * self.relativistic_velocity(p_half)
        f_curr = self.total_conservative_force(q_t, top_idx, gates)
        p_t = df * p_half - hdt * f_curr + hdt * f_ext
        return q_t, p_t, c_t, top_idx, gates, flux, mdl_cost


class ArjunaLieGroupDiscoveryCore(nn.Module):
    """Arjuna v15.0 Core with Autonomous Phase Clock & Lie-Group Shift Factorization."""

    def __init__(
        self,
        vocab_size: int = 48,
        dim: int = 256,
        embed_rank: int = 64,
        block_size: int = 64,
        num_coarse_mem: int = 32,
        num_leaf_mem: int = 32,
        num_nodes: int = 16,
        op_rank: int = 64,
        top_k: int = 2,
        num_lie_shifts: int = 8,
        beta_pc: float = 0.05,
        active_ring_roll: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed_rank = embed_rank
        self.block_size = block_size
        self.num_lie_shifts = num_lie_shifts
        self.beta_pc = beta_pc
        self.active_ring_roll = active_ring_roll

        q_emb_t, _ = torch.linalg.qr(torch.randn(embed_rank, vocab_size))
        self.tok_embed = nn.Parameter(q_emb_t.t().contiguous() * math.sqrt(float(embed_rank)), requires_grad=False)

        q_proj_t, _ = torch.linalg.qr(torch.randn(dim, embed_rank))
        self.tok_proj = nn.Parameter(q_proj_t.t().contiguous() * math.sqrt(float(dim / embed_rank)), requires_grad=False)

        self.binder = ArjunaBlockHadamardALU(dim, block_size)
        self.world_memory = ArjunaHolographicCartridge(dim, num_coarse_mem, num_leaf_mem)
        self.cell = SymplecticRelativisticCell(dim, num_nodes, op_rank, top_k)

        scale_d = 1.0 / math.sqrt(float(dim))
        self.w_alu_3b = nn.Parameter(torch.eye(dim) + 0.02 * torch.randn(dim, dim) * scale_d, requires_grad=False)
        self.w_mom = nn.Parameter(torch.randn(dim, dim) * scale_d, requires_grad=False)
        self.w_out = nn.Parameter(torch.zeros(vocab_size, dim), requires_grad=False)
        self.w_invariant = nn.Parameter(torch.zeros(num_lie_shifts, dim), requires_grad=False)
        self.readout_norm = nn.Parameter(torch.ones(dim), requires_grad=False)

    def normalize_manifold(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + 1e-5)).to(dtype=x.dtype) * self.readout_norm.to(dtype=x.dtype)

    @staticmethod
    def phase_rotate_right(x: torch.Tensor, shift_blocks: int = 1) -> torch.Tensor:
        return torch.roll(x, shifts=shift_blocks * 64, dims=-1)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
        z_d = lambda: torch.zeros(batch_size, self.dim, device=device, dtype=dtype)
        c_0 = torch.full((batch_size, self.cell.router.num_nodes), 1.0 / float(self.cell.router.num_nodes), device=device, dtype=dtype)
        tau_0 = torch.zeros(batch_size, dtype=torch.long, device=device)
        return z_d(), z_d(), c_0, z_d(), z_d(), z_d(), tau_0

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        compute_local_updates: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            bsz, seq_len = input_ids.shape
            dev = input_ids.device
            dtp = torch.float16 if dev.type == "cuda" else torch.float32
            q_t, p_t, c_t, r_a_t, r_op_t, s_carry_t, tau_t = self.init_state(bsz, dev, dtp)

            te = self.tok_embed.to(dtype=dtp)
            tp = self.tok_proj.to(dtype=dtp)
            wa = self.w_alu_3b.to(dtype=dtp)
            wm = self.w_mom.to(dtype=dtp)
            wo = self.w_out.to(dtype=dtp)
            wi = self.w_invariant.to(dtype=dtp)

            if compute_local_updates and target_ids is not None:
                d_wa = torch.zeros_like(self.w_alu_3b, dtype=torch.float32)
                d_wm = torch.zeros_like(self.w_mom, dtype=torch.float32)
                d_wo = torch.zeros_like(self.w_out, dtype=torch.float32)
                d_wi = torch.zeros_like(self.w_invariant, dtype=torch.float32)
                d_ou = torch.zeros_like(self.cell.router.op_u, dtype=torch.float32)
                d_ov = torch.zeros_like(self.cell.router.op_v, dtype=torch.float32)
                d_ck = torch.zeros_like(self.world_memory.coarse_keys, dtype=torch.float32)
                d_lk = torch.zeros_like(self.world_memory.leaf_keys, dtype=torch.float32)
                d_lv = torch.zeros_like(self.world_memory.leaf_values, dtype=torch.float32)

            t_ce = torch.zeros(1, device=dev, dtype=torch.float32)
            t_pc = torch.zeros(1, device=dev, dtype=torch.float32)
            t_mdl = torch.zeros(1, device=dev, dtype=torch.float32)
            t_en = torch.zeros(1, device=dev, dtype=torch.float32)
            t_ok = torch.zeros(1, device=dev, dtype=torch.float32)
            t_ev = torch.zeros(1, device=dev, dtype=torch.float32)
            c_hits = torch.zeros(1, device=dev, dtype=torch.float32)
            prev_top: Optional[torch.Tensor] = None
            logits_list: List[torch.Tensor] = []

            reward_mod = torch.ones(bsz, self.cell.router.num_nodes, device=dev, dtype=torch.float32)
            base_l = torch.tensor(3.87, device=dev, dtype=torch.float32)

            for t in range(seq_len):
                x_t = input_ids[:, t]
                q_prev, p_prev, c_prev, s_prev = q_t, p_t, c_t, s_carry_t
                u_t = torch.matmul(te[x_t], tp)
                act = x_t != 0

                ia = (act & (tau_t == 0)).to(dtype=dtp).unsqueeze(-1)
                iop = (act & (tau_t == 1)).to(dtype=dtp).unsqueeze(-1)
                ib = (act & (tau_t == 2)).to(dtype=dtp).unsqueeze(-1)

                r_a_t = (1.0 - ia) * r_a_t + ia * u_t
                r_op_t = (1.0 - iop) * r_op_t + iop * u_t

                w2 = self.binder(r_a_t, self.phase_rotate_right(r_op_t, 1))
                w3 = self.binder(w2, self.phase_rotate_right(u_t, 2))
                alu_3b = torch.matmul(w3, wa)

                q_mem = w3 + s_prev
                m_t, c_star, s_fine = self.world_memory.retrieve(q_mem)

                f_ext = ArjunaBlockHadamardALU.algebraic_saturate(alu_3b + 0.20 * m_t)
                q_t, p_t, c_t, top_idx, gates, _, mdl_c = self.cell(q_prev, p_prev, c_prev, f_ext, reward_mod)

                if prev_top is not None:
                    c_hits += (top_idx[:, 0] == prev_top[:, 0]).float().mean()
                prev_top = top_idx

                h_out = self.normalize_manifold(alu_3b + 0.15 * q_t + 0.15 * torch.matmul(p_t, wm) + 0.15 * m_t)
                l_base = torch.matmul(h_out, wo.t()).float() * 8.0
                if self.active_ring_roll:
                    l_base[:, 0] = -1e9

                s_norm = F.normalize(s_prev.float(), dim=-1)
                has_c = (torch.sum(s_prev.float() ** 2, dim=-1, keepdim=True) > 1e-4).float()
                inv_l = torch.matmul(s_norm, wi.float().t()) * 6.0
                z_pr = torch.zeros_like(inv_l)
                z_pr[:, 0] = 20.0
                pi_c = F.softmax(has_c * inv_l + (1.0 - has_c) * z_pr, dim=-1)

                p_base = F.softmax(l_base, dim=-1)
                if self.active_ring_roll:
                    pr, pz = p_base[:, 1:], p_base[:, :1]
                    s_stack = torch.stack([torch.cat([pz, torch.roll(pr, m, dims=-1)], dim=-1) for m in range(self.num_lie_shifts)], dim=1)
                else:
                    s_stack = torch.stack([torch.roll(p_base, m, dims=-1) for m in range(self.num_lie_shifts)], dim=1)

                p_fin = torch.sum(pi_c.unsqueeze(-1) * s_stack, dim=1)
                l_fin = torch.log(torch.clamp(p_fin, min=1e-7))

                if not compute_local_updates:
                    logits_list.append(l_fin)

                s_carry_t = (1.0 - ib) * s_prev + ib * self.binder(r_a_t, self.phase_rotate_right(u_t, 1))
                tau_t = torch.where(act, (tau_t + 1) % 4, tau_t)

                t_en += torch.mean(torch.sqrt(1.0 + torch.mean(p_t.float() ** 2, dim=-1)) + 0.25 * torch.sqrt(1.0 + torch.mean(q_t.float() ** 2, dim=-1)))
                t_mdl += mdl_c

                if target_ids is not None:
                    y_t = target_ids[:, t]
                    sm = loss_mask[:, t].float() if loss_mask is not None else torch.ones(bsz, device=dev, dtype=torch.float32)
                    msum = torch.sum(sm)
                    pt_ce = F.nll_loss(l_fin, y_t, reduction="none")

                    if msum > 0:
                        mce = torch.sum(pt_ce * sm)
                        t_ce += mce
                        t_ok += torch.sum((torch.argmax(l_fin, dim=-1) == y_t).float() * sm)
                        t_ev += msum
                        radv = torch.clamp(base_l - pt_ce.detach(), -1.5, 1.5)
                        base_l = 0.95 * base_l + 0.05 * (mce / msum).detach()
                        reward_mod = torch.ones(bsz, self.cell.router.num_nodes, device=dev, dtype=torch.float32)
                        reward_mod.scatter_(1, top_idx, torch.exp(0.30 * radv).unsqueeze(-1).expand(-1, self.cell.router.top_k))

                    pc_d = (q_t - (q_prev + self.cell.dt * self.cell.relativistic_velocity(p_prev))).float()
                    t_pc += 0.5 * torch.mean(pc_d * pc_d)

                    if compute_local_updates and msum > 0:
                        isc = 8.0 / float(max(1, int(msum.item())))
                        oh_y = F.one_hot(y_t, num_classes=self.vocab_size).float()
                        yg = y_t.view(bsz, 1, 1).expand(bsz, self.num_lie_shifts, 1)
                        slik = torch.gather(s_stack, -1, yg).squeeze(-1)
                        upost = pi_c * slik + 1e-8
                        gpost = upost / torch.sum(upost, dim=-1, keepdim=True)

                        d_wi.add_(torch.matmul(((pi_c - gpost) * sm.unsqueeze(-1) * has_c * isc).t(), s_norm))

                        utarg = torch.zeros_like(oh_y)
                        if self.active_ring_roll:
                            roh, zoh = oh_y[:, 1:], oh_y[:, :1]
                            for m in range(self.num_lie_shifts):
                                utarg.add_(gpost[:, m : m + 1] * torch.cat([zoh, torch.roll(roh, -m, dims=-1)], dim=-1))
                        else:
                            for m in range(self.num_lie_shifts):
                                utarg.add_(gpost[:, m : m + 1] * torch.roll(oh_y, -m, dims=-1))

                        eps_b = (p_base - utarg) * sm.unsqueeze(-1) * isc
                        d_wo.add_(torch.matmul(eps_b.t(), h_out.float()))
                        dh = torch.matmul(eps_b, self.w_out.float())
                        dq = 0.15 * dh + (self.beta_pc * isc) * pc_d
                        dp = 0.15 * torch.matmul(dh, self.w_mom.float().t())

                        d_wm.add_(torch.matmul(p_t.float().t(), 0.15 * dh))
                        d_wa.add_(torch.matmul(w3.float().t(), dh))

                        du_s, dv_s = self.cell.router.compute_local_operator_plasticity(0.5 * (q_prev + q_t), top_idx, gates, -0.5 * self.cell.dt * (dq + dp))
                        d_ou.add_(du_s)
                        d_ov.add_(dv_s)

                        dck, dlk, dlv = self.world_memory.compute_local_plasticity(q_mem, c_star, s_fine, 0.15 * dh)
                        d_ck.add_(dck)
                        d_lk.add_(dlk)
                        d_lv.add_(dlv)

            den = torch.clamp(t_ev, min=1.0)
            if compute_local_updates and target_ids is not None:
                return (
                    (t_ce / den).unsqueeze(0), (t_pc / seq_len).unsqueeze(0), (t_mdl / seq_len).unsqueeze(0),
                    (t_en / seq_len).unsqueeze(0), (t_ok / den).unsqueeze(0), (c_hits / max(1, seq_len - 1)).unsqueeze(0),
                    d_wa.unsqueeze(0), d_wm.unsqueeze(0), d_wo.unsqueeze(0), d_wi.unsqueeze(0),
                    d_ou.unsqueeze(0), d_ov.unsqueeze(0), d_ck.unsqueeze(0), d_lk.unsqueeze(0), d_lv.unsqueeze(0),
                )
            return (
                torch.stack(logits_list, dim=1) if logits_list else torch.empty(0, device=dev),
                (t_ce / den).unsqueeze(0), (t_pc / seq_len).unsqueeze(0), (t_mdl / seq_len).unsqueeze(0),
                (t_en / seq_len).unsqueeze(0), (t_ok / den).unsqueeze(0), (c_hits / max(1, seq_len - 1)).unsqueeze(0),
            )


class ArjunaLieGroupOptimizer:
    def __init__(self, model: ArjunaLieGroupDiscoveryCore, lr: float = 2.5e-2, weight_decay: float = 1e-4):
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_count = 0
        self.param_list = [
            model.w_alu_3b, model.w_mom, model.w_out, model.w_invariant,
            model.cell.router.op_u, model.cell.router.op_v,
            model.world_memory.coarse_keys, model.world_memory.leaf_keys, model.world_memory.leaf_values,
        ]
        self.m = [torch.zeros_like(p, dtype=torch.float32) for p in self.param_list]
        self.v = [torch.zeros_like(p, dtype=torch.float32) for p in self.param_list]

    def apply_local_deltas(self, deltas: Sequence[torch.Tensor]) -> None:
        with torch.no_grad():
            self.step_count += 1
            bc1, bc2 = 1.0 - 0.9 ** self.step_count, 1.0 - 0.99 ** self.step_count
            for i, (p, d) in enumerate(zip(self.param_list, deltas)):
                g = torch.mean(d.float(), dim=0)
                self.m[i].mul_(0.9).add_(g, alpha=0.1)
                self.v[i].mul_(0.99).addcmul_(g, g, value=0.01)
                if self.weight_decay > 0.0 and i < 6:
                    p.data.mul_(1.0 - self.lr * self.weight_decay)
                p.data.add_(((self.m[i] / bc1) / (torch.sqrt(self.v[i] / bc2) + 1e-8)).to(dtype=p.dtype), alpha=-self.lr)


def generate_universe_stream(u_id: int, bsz: int, n_clauses: int, vocab_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = n_clauses * 5
    pmod = vocab_size - 1
    ka = torch.randint(1, 5, (bsz, n_clauses), device=device)
    op = torch.randint(1, 4, (bsz, n_clauses), device=device)
    kb = torch.randint(1, 5, (bsz, n_clauses), device=device)
    tokens = torch.zeros(bsz, seq_len + 1, dtype=torch.long, device=device)
    mask = torch.zeros(bsz, seq_len, dtype=torch.float32, device=device)
    carry = torch.zeros(bsz, dtype=torch.long, device=device)
    for c in range(n_clauses):
        base = c * 5
        a_c, op_c, b_c = ka[:, c], op[:, c], kb[:, c]
        if u_id == 1:
            res = (a_c * op_c + b_c + carry) % vocab_size
            carry = (a_c + b_c) % 3
        elif u_id == 2:
            res = 1 + ((((a_c * a_c + op_c * (b_c * b_c)) % pmod) + carry - 1) % pmod)
            carry = (a_c * b_c) % 4
        else:
            res = 1 + (((((a_c ** 3) + (op_c ** 2) * b_c) % pmod) + carry - 1) % pmod)
            carry = (a_c ^ b_c) % 8
        tokens[:, base], tokens[:, base + 1], tokens[:, base + 2], tokens[:, base + 3], tokens[:, base + 4] = a_c, 0, op_c, b_c, res
        mask[:, base + 3] = 1.0
    return tokens[:, :-1], tokens[:, 1:], mask


if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Running Project Arjuna (T-PHPN v15.0) Reference Suite on {dev} ({ngpu} GPUs)...")
    for uid, name in [(1, "Linear-Z3"), (2, "Quadratic-Z4"), (3, "Cubic-XOR-Z8")]:
        m = ArjunaLieGroupDiscoveryCore(active_ring_roll=(uid >= 2)).to(dev)
        pm = nn.DataParallel(m, device_ids=[0, 1]) if ngpu >= 2 else m
        opt = ArjunaLieGroupOptimizer(m, lr=2.6e-2)
        for s in range(1, 81):
            if s > 50:
                opt.lr = 1e-3 + 0.5 * (2.6e-2 - 1e-3) * (1.0 + math.cos(math.pi * (s - 50) / 30.0))
            xb, yb, mb = generate_universe_stream(uid, 128 if ngpu >= 2 else 64, 10, 48, dev)
            out = pm(xb, yb, mb, compute_local_updates=True)
            opt.apply_local_deltas(out[6:])
        xv, yv, mv = generate_universe_stream(uid, 256, 12, 48, dev)
        vo = m(xv, yv, mv, compute_local_updates=False)
        print(f"  -> {name:15s} | Unseen Accuracy: {vo[5].item()*100.0:6.2f}% | CE: {vo[1].item():.4f}")
