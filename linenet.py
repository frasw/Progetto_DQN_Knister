"""LineNet: rete dueling con scorer di linea condiviso per Knister.

Questo modulo definisce l'architettura LineNet, il ramo di encoding dello stato
che la alimenta e i validatori di correttezza. Non modifica
train_knister_v2.py: lo importa per riusarne le primitive (score_lines_torch,
FastVectorKnister, DEFAULT_SCORE_TABLE), così la logica di punteggio resta
un'unica fonte di verità condivisa tra rete e ambiente.

USO
---
    python linenet.py                # esegue i validatori su CPU
    python linenet.py --device cuda  # ripete i controlli su GPU

MOTIVAZIONE
-----------
Un MLP su un vettore piatto di feature tratta le 25 uscite come indipendenti e
deve riapprendere da zero, per ogni zona della griglia, la stessa regola di
punteggio. Il punteggio di Knister si fattorizza invece come somma su 12 linee
(5 righe, 5 colonne, 2 diagonali) della medesima funzione di combinazione, con
un peso maggiore sulle diagonali. LineNet incorpora questa struttura: una sola
rete "scorer" viene applicata con gli stessi pesi a tutte le linee, e ciò che
viene appreso su una riga vale automaticamente per ogni colonna e diagonale.

CONTRATTO
---------
- Stato "line": tensore float32 [N, 34] = [griglia(25), lancio(1), tabella(8)],
  prodotto da encode_line_states(states_u8, device, score_table). I valori non
  sono normalizzati: la rete ricostruisce gli interi internamente.
- LineNet.forward(x, valid_mask=None) -> Q [N, 25]. valid_mask è accettata per
  compatibilità di interfaccia con DQN ma ignorata: la maschera di validità
  viene derivata internamente da (griglia == 0), che coincide con quella
  calcolata dai chiamanti. La media dueling è sempre mascherata sulle sole
  azioni valide.
- LineNet.immediate_rewards(x) -> [N, 25]: reward immediato esatto per ogni
  azione valida (0 sulle celle occupate). Per costruzione coincide con il
  reward dell'ambiente.

ARCHITETTURA (default: embed=64, ctx=128, head=256 -> 110.786 parametri)
------------------------------------------------------------------------
Per ciascuna delle 12 linee:
    e_l = phi(istogramma valori/5, n_riempiti/5, flag_diag, peso/2,
              punteggio_linea_pesato/C, tabella/C)          [phi condivisa]
Per ciascuna delle 60 incidenze (cella, linea, slot):
    what-if con il lancio corrente inserito nello slot, da cui
    e'_l e Delta_s = punteggio_pesato(dopo) - punteggio_pesato(prima)
Per azione c:  a_c = [somma(e'-e), somma(e'), Delta_r_c/C_imm, n_linee(c)/4]
Contesto:      ctx = ReLU(W . [media(e), max(e), globali, tabella/C])
Teste dueling: V = MLP(ctx);  A_c = psi([a_c, ctx]) condivisa sulle 25 celle;
               Q = V + A - media_mascherata(A)

Le feature non contengono alcuna identità di riga, colonna o posizione (solo
il flag diagonale e il peso): di conseguenza Q è esattamente equivariante
rispetto alle 8 simmetrie del gruppo diedrale D4, per costruzione nei pesi e
non per addestramento. La costante C di normalizzazione è il massimo della
tabella al momento della costruzione, salvata come buffer nel checkpoint.

VALIDATORI
----------
  1. encoding "line": shape, dtype, passthrough, tabella default e per-sample
  2. decomposizione del reward: immediate_rewards coincide con il reward di
     FastVectorKnister su rollout completi, con tabella default e campionate
  3. equivarianza D4 esatta delle Q, con permutazioni costruite localmente e
     indipendenti dal codice di augmentation del motore
  4. forma e finitezza (stato iniziale, intermedio, terminale con lancio=0)
     più un passo di backward con gradienti finiti
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Import robusto del modulo base (train_knister_v2.py, anche con suffissi tipo
# train_knister_v2-1.py scaricati dal browser).
# ----------------------------------------------------------------------------
def _load_base_module():
    try:
        import train_knister_v2 as tk  # type: ignore
        return tk
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in sorted(glob.glob(os.path.join(here, "train_knister_v2*.py"))):
            spec = importlib.util.spec_from_file_location("train_knister_v2", cand)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["train_knister_v2"] = mod
            spec.loader.exec_module(mod)
            print(f"[linenet] modulo base caricato da: {os.path.basename(cand)}")
            return mod
        raise ImportError(
            "train_knister_v2.py non trovato nella cartella di linenet.py"
        )


tk = _load_base_module()

try:
    from api import KnisterGame  # stessa fonte immutabile usata da v2
except ImportError as exc:  # pragma: no cover
    raise ImportError("api.py non trovato accanto a linenet.py") from exc

DIAG_MULT = float(KnisterGame.DIAGONAL_MULTIPLIER)

# Dimensione dello stato "line": griglia(25) + lancio(1) + tabella(8)
LINE_STATE_SIZE = 34
_N_CELLS = 25
_N_LINES = 12
_N_INC = 60
_PHI_IN = 11 + 1 + 1 + 1 + 1 + 8  # 23


# ----------------------------------------------------------------------------
# Geometria del gioco: linee, incidenze (cella, linea, slot), appartenenze.
# ----------------------------------------------------------------------------
def build_line_geometry():
    """Costruisce gli indici di linea/incidenza come array numpy.

    Ritorna un dict con:
      line_cells   [12, 5]  indice di cella piatta per ogni slot di ogni linea
      line_is_diag [12]     1.0 per le due diagonali, 0.0 altrimenti
      line_weight  [12]     1.0 per righe/colonne, DIAG_MULT per le diagonali
      inc_cell     [60]     cella di ciascuna incidenza
      inc_line     [60]     linea di ciascuna incidenza
      inc_slot     [60]     slot (0..4) della cella dentro la linea
      cell_membership [25]  numero di linee passanti per la cella (2, 3 o 4)
    """
    line_cells = np.zeros((_N_LINES, 5), dtype=np.int64)
    for r in range(5):  # righe 0..4
        line_cells[r] = [5 * r + c for c in range(5)]
    for c in range(5):  # colonne 5..9
        line_cells[5 + c] = [5 * r + c for r in range(5)]
    line_cells[10] = [0, 6, 12, 18, 24]   # diagonale principale
    line_cells[11] = [4, 8, 12, 16, 20]   # anti-diagonale

    line_is_diag = np.zeros(_N_LINES, dtype=np.float32)
    line_is_diag[10:] = 1.0
    line_weight = np.where(line_is_diag > 0, DIAG_MULT, 1.0).astype(np.float32)

    inc_cell, inc_line, inc_slot = [], [], []
    for cell in range(_N_CELLS):
        r, c = divmod(cell, 5)
        members = [(r, c), (5 + c, r)]
        if r == c:
            members.append((10, r))
        if r + c == 4:
            members.append((11, r))
        for line, slot in members:
            inc_cell.append(cell)
            inc_line.append(line)
            inc_slot.append(slot)

    inc_cell = np.asarray(inc_cell, dtype=np.int64)
    inc_line = np.asarray(inc_line, dtype=np.int64)
    inc_slot = np.asarray(inc_slot, dtype=np.int64)
    assert inc_cell.shape[0] == _N_INC

    # coerenza: LINE_CELLS[inc_line, inc_slot] == inc_cell
    assert np.array_equal(line_cells[inc_line, inc_slot], inc_cell)

    cell_membership = np.bincount(inc_cell, minlength=_N_CELLS).astype(np.float32)
    assert cell_membership.min() == 2 and cell_membership.max() == 4

    return {
        "line_cells": line_cells,
        "line_is_diag": line_is_diag,
        "line_weight": line_weight,
        "inc_cell": inc_cell,
        "inc_line": inc_line,
        "inc_slot": inc_slot,
        "cell_membership": cell_membership,
    }


# ----------------------------------------------------------------------------
# Ramo encoder per state_encoding == "line".
# ----------------------------------------------------------------------------
def encode_line_states(states_u8, device, score_table=None):
    """Codifica stati grezzi [N, 26] uint8 nello stato "line" [N, 34] float32.

    Nessuna normalizzazione: [griglia(25), lancio(1)] passano tali e quali,
    seguiti dalla tabella punteggi (8 valori grezzi). score_table puo' essere:
      - None            -> tk.DEFAULT_SCORE_TABLE condivisa da tutto il batch
      - array/tensore [8]      -> condivisa da tutto il batch
      - tensore [N, 8]         -> per-campione (replay con randomize_scores)
    """
    states = np.ascontiguousarray(states_u8, dtype=np.uint8)
    if states.ndim != 2 or states.shape[1] != 26:
        raise ValueError(f"attesi stati [N, 26], ricevuto {states.shape}")
    n = states.shape[0]
    raw = torch.from_numpy(states).to(device=device, dtype=torch.float32)

    if score_table is None:
        score_table = tk.DEFAULT_SCORE_TABLE
    table = torch.as_tensor(score_table, dtype=torch.float32, device=device)
    if table.dim() == 1:
        if table.shape[0] != 8:
            raise ValueError(f"tabella condivisa attesa [8], ricevuta {tuple(table.shape)}")
        table = table.view(1, 8).expand(n, 8)
    elif table.dim() == 2:
        if table.shape != (n, 8):
            raise ValueError(
                f"tabella per-campione attesa [{n}, 8], ricevuta {tuple(table.shape)}"
            )
    else:
        raise ValueError("score_table deve avere 1 o 2 dimensioni")

    return torch.cat([raw, table], dim=1)


def line_state_input_size():
    """Dimensione del vettore di stato per l'encoding 'line': 34."""
    return LINE_STATE_SIZE


# ----------------------------------------------------------------------------
# LineNet
# ----------------------------------------------------------------------------
class LineNet(nn.Module):
    """Dueling DQN strutturata: scorer di linea condiviso + testa per azione.

    forward(x, valid_mask=None) -> Q [N, 25]
      x: stato "line" [N, 34] float32 (vedi encode_line_states).
      valid_mask: ignorata (compatibilita' di firma con DQN); la maschera e'
      derivata internamente dalla griglia e la media dueling e' sempre
      mascherata sulle sole azioni valide.
    """

    def __init__(self, embed_dim: int = 64, ctx_dim: int = 128,
                 head_dim: int = 256, score_ceiling: float | None = None):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.ctx_dim = int(ctx_dim)
        self.head_dim = int(head_dim)

        geo = build_line_geometry()
        self.register_buffer("line_cells", torch.from_numpy(geo["line_cells"]))
        self.register_buffer("line_is_diag", torch.from_numpy(geo["line_is_diag"]))
        self.register_buffer("line_weight", torch.from_numpy(geo["line_weight"]))
        self.register_buffer("inc_cell", torch.from_numpy(geo["inc_cell"]))
        self.register_buffer("inc_line", torch.from_numpy(geo["inc_line"]))
        self.register_buffer("inc_slot", torch.from_numpy(geo["inc_slot"]))
        self.register_buffer("cell_membership", torch.from_numpy(geo["cell_membership"]))
        self.register_buffer(
            "dice_probs",
            torch.as_tensor(np.asarray(tk.DICE_SUM_PROBABILITIES, dtype=np.float32)),
        )

        # Pin della normalizzazione: massimo della tabella al momento della
        # costruzione, salvato nello state_dict (sopravvive a modifiche di api.py).
        if score_ceiling is None:
            score_ceiling = float(np.asarray(tk.DEFAULT_SCORE_TABLE, dtype=np.float32).max())
        self.register_buffer("score_ceiling", torch.tensor(float(score_ceiling)))

        e, c, h = self.embed_dim, self.ctx_dim, self.head_dim
        self.phi = nn.Sequential(
            nn.Linear(_PHI_IN, e), nn.ReLU(),
            nn.Linear(e, e), nn.ReLU(),
        )
        self.ctx_lin = nn.Linear(2 * e + 42, c)
        self.value_head = nn.Sequential(
            nn.Linear(c, c), nn.ReLU(),
            nn.Linear(c, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(2 * e + 2 + c, h), nn.ReLU(),
            nn.Linear(h, 1),
        )

    # -- utilita' ------------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _phi_input(self, hist, n_filled, is_diag, weight, score_w, table_n):
        """Assembla l'input [., K, 23] di phi (K = 12 linee o 60 incidenze)."""
        n, k = hist.shape[0], hist.shape[1]
        return torch.cat(
            [
                hist / 5.0,
                (n_filled / 5.0).unsqueeze(-1),
                is_diag.view(1, k, 1).expand(n, k, 1),
                (weight / DIAG_MULT).view(1, k, 1).expand(n, k, 1),
                (score_w / self.score_ceiling).unsqueeze(-1),
                table_n.unsqueeze(1).expand(n, k, 8),
            ],
            dim=-1,
        )

    @staticmethod
    def _hist_and_filled(vals):
        """vals long [., K, 5] -> (istogramma [., K, 11], n_riempiti [., K])."""
        hist = F.one_hot(vals.clamp(min=0, max=12), num_classes=13)[..., 2:]
        return hist.sum(dim=2).float(), vals.ne(0).sum(dim=2).float()

    # -- nucleo condiviso -----------------------------------------------------
    def _core(self, x):
        if x.dim() != 2 or x.shape[1] != LINE_STATE_SIZE:
            raise ValueError(
                f"LineNet si aspetta input [N, {LINE_STATE_SIZE}], ricevuto {tuple(x.shape)}"
            )
        n = x.shape[0]
        grid = x[:, :_N_CELLS].round().long()          # [N, 25] valori 0..12
        roll = x[:, _N_CELLS].round().long()           # [N]     2..12 (0 = terminale)
        table = x[:, _N_CELLS + 1:]                    # [N, 8]  float
        valid = grid.eq(0)                             # [N, 25] bool
        table_n = table / self.score_ceiling

        # ---- linee di base ----
        line_vals = grid[:, self.line_cells]                       # [N, 12, 5]
        base_raw = tk.score_lines_torch(line_vals, table)          # [N, 12]
        base_w = base_raw * self.line_weight                       # [N, 12]
        hist, n_filled = self._hist_and_filled(line_vals)
        e_base = self.phi(
            self._phi_input(hist, n_filled, self.line_is_diag,
                            self.line_weight, base_w, table_n)
        )                                                          # [N, 12, E]

        # ---- what-if per incidenza (cella, linea, slot) ----
        idx = self.inc_slot.view(1, _N_INC, 1).expand(n, _N_INC, 1)
        src = roll.view(n, 1, 1).expand(n, _N_INC, 1)
        hyp_vals = line_vals[:, self.inc_line, :].scatter(2, idx, src)  # [N, 60, 5]
        hyp_raw = tk.score_lines_torch(hyp_vals, table)                 # [N, 60]
        hyp_w = hyp_raw * self.line_weight[self.inc_line]
        delta_s = hyp_w - base_w[:, self.inc_line]                      # [N, 60]
        hyp_hist, hyp_filled = self._hist_and_filled(hyp_vals)
        e_hyp = self.phi(
            self._phi_input(hyp_hist, hyp_filled,
                            self.line_is_diag[self.inc_line],
                            self.line_weight[self.inc_line], hyp_w, table_n)
        )                                                               # [N, 60, E]
        delta_e = e_hyp - e_base[:, self.inc_line]                      # [N, 60, E]

        # ---- pulizia: azzera i contributi delle incidenze su celle occupate ----
        inc_valid = valid[:, self.inc_cell]                             # [N, 60]
        vmask = inc_valid.to(x.dtype).unsqueeze(-1)
        delta_e = delta_e * vmask
        e_hyp_m = e_hyp * vmask
        delta_s = delta_s * inc_valid.to(x.dtype)

        # ---- aggregazione per azione ----
        e_dim = self.embed_dim
        sum_de = x.new_zeros((n, _N_CELLS, e_dim)).index_add_(1, self.inc_cell, delta_e)
        sum_eh = x.new_zeros((n, _N_CELLS, e_dim)).index_add_(1, self.inc_cell, e_hyp_m)
        r_all = x.new_zeros((n, _N_CELLS)).index_add_(1, self.inc_cell, delta_s)

        return {
            "grid": grid, "roll": roll, "valid": valid, "table_n": table_n,
            "e_base": e_base, "sum_de": sum_de, "sum_eh": sum_eh, "r_all": r_all,
        }

    # -- API pubbliche ---------------------------------------------------------
    @torch.no_grad()
    def immediate_rewards(self, x):
        """Reward immediato esatto per azione [N, 25]; 0 sulle celle occupate."""
        return self._core(x)["r_all"]

    def forward(self, x, valid_mask=None):  # valid_mask ignorata (vedi docstring)
        core = self._core(x)
        n = x.shape[0]
        grid, roll, valid = core["grid"], core["roll"], core["valid"]

        # ---- contesto globale ----
        roll_idx = (roll - 2).clamp(min=0, max=10)
        roll_ok = ((roll >= 2) & (roll <= 12)).to(x.dtype).unsqueeze(1)
        roll_oh = F.one_hot(roll_idx, num_classes=11).to(x.dtype) * roll_ok   # [N, 11]

        board_counts = F.one_hot(grid.clamp(min=0, max=12), num_classes=13)[..., 2:]
        board_counts = board_counts.sum(dim=1).to(x.dtype)                    # [N, 11]
        filled = board_counts.sum(dim=1, keepdim=True)                        # [N, 1]
        remaining = 25.0 - filled
        forecast_n = (board_counts + remaining * self.dice_probs.view(1, 11)) / 25.0

        ctx_in = torch.cat(
            [
                core["e_base"].mean(dim=1),
                core["e_base"].max(dim=1).values,
                roll_oh,
                filled / 25.0,
                board_counts / 25.0,
                forecast_n,
                core["table_n"],
            ],
            dim=1,
        )                                                                     # [N, 2E+42]
        ctx = F.relu(self.ctx_lin(ctx_in))                                    # [N, C]

        # ---- teste dueling ----
        v = self.value_head(ctx)                                              # [N, 1]
        imm_ceiling = self.score_ceiling * (2.0 + 2.0 * DIAG_MULT)
        scalars = torch.stack(
            [
                core["r_all"] / imm_ceiling,
                (self.cell_membership / 4.0).view(1, _N_CELLS).expand(n, _N_CELLS),
            ],
            dim=-1,
        )                                                                     # [N, 25, 2]
        psi_in = torch.cat(
            [
                core["sum_de"],
                core["sum_eh"],
                scalars,
                ctx.unsqueeze(1).expand(n, _N_CELLS, self.ctx_dim),
            ],
            dim=-1,
        )
        adv = self.adv_head(psi_in).squeeze(-1)                               # [N, 25]

        vf = valid.to(x.dtype)
        cnt = vf.sum(dim=1, keepdim=True).clamp(min=1.0)
        adv_mean = (adv * vf).sum(dim=1, keepdim=True) / cnt
        return v + adv - adv_mean


# ============================================================================
# VALIDATORI — python linenet.py [--device cpu|cuda] [--seed 123]
# ============================================================================
def _make_env(n_envs, rng, score_table=None):
    if score_table is None:
        return tk.FastVectorKnister(n_envs, rng)
    try:
        return tk.FastVectorKnister(n_envs, rng, score_table=score_table)
    except TypeError:
        return tk.FastVectorKnister(n_envs, rng, score_table)


def _step_env(env, actions):
    """Estrae i reward per-ambiente da un passo di FastVectorKnister.

    Il contratto di step(actions) è (next_state, reward, done): il reward è
    quindi il secondo elemento. La guardia sulla shape fa fallire in modo
    esplicito se il contratto cambiasse, invece di confrontare silenziosamente
    l'array sbagliato.
    """
    out = env.step(actions)
    rewards = out[1] if isinstance(out, (tuple, list)) else out
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 1:
        raise RuntimeError(
            f"attesi reward 1-D per-env da step(), ricevuta shape {rewards.shape}: "
            "il contratto (next_state, reward, done) e' cambiato?"
        )
    return rewards


def _sampled_table(rng):
    try:
        return np.asarray(tk.sample_score_table(rng, 0.5, 2.0), dtype=np.float32)
    except (AttributeError, TypeError):
        base = np.asarray(tk.DEFAULT_SCORE_TABLE, dtype=np.float32)
        jitter = rng.uniform(0.5, 2.0, size=base.shape).astype(np.float32)
        return np.maximum(np.rint(base * jitter), 1.0).astype(np.float32)


def _build_d4_perms():
    """Permutazioni D4 costruite localmente (indipendenti da v2).

    perm[g, k] = indice della cella ORIGINALE che finisce in posizione k della
    griglia trasformata; inv[g] = argsort(perm[g]) mappa cella originale ->
    posizione trasformata.
    """
    idx = np.arange(25).reshape(5, 5)
    grids = []
    for rot in range(4):
        g = np.rot90(idx, rot)
        grids.append(g)
        grids.append(np.fliplr(g))
    perm = np.stack([g.reshape(-1) for g in grids]).astype(np.int64)   # [8, 25]
    inv = np.argsort(perm, axis=1).astype(np.int64)                    # [8, 25]
    return perm, inv


def validate_line_encoding(seed=123, device="cpu"):
    rng = np.random.default_rng(seed)
    env = _make_env(16, rng)
    for _ in range(rng.integers(3, 12)):
        states = env.observe()
        valid = states[:, :25] == 0
        _step_env(env, tk.random_valid_actions(valid, rng))
    states = env.observe()
    n = states.shape[0]

    x = encode_line_states(states, device)
    assert x.shape == (n, LINE_STATE_SIZE) and x.dtype == torch.float32
    assert np.array_equal(
        x[:, :26].cpu().numpy().astype(np.uint8), states
    ), "passthrough griglia+lancio non esatto"
    default = np.asarray(tk.DEFAULT_SCORE_TABLE, dtype=np.float32)
    assert np.allclose(x[:, 26:].cpu().numpy(), np.tile(default, (n, 1))), \
        "tabella default non replicata correttamente"

    per_sample = np.stack([_sampled_table(rng) for _ in range(n)]).astype(np.float32)
    x2 = encode_line_states(states, device, torch.from_numpy(per_sample).to(device))
    assert np.allclose(x2[:, 26:].cpu().numpy(), per_sample), \
        "tabella per-campione non rispettata"
    print("[OK] 1/4 encoding 'line': shape, dtype, passthrough, tabelle")


def validate_reward_decomposition(seed=123, device="cpu", n_envs=64, atol=1e-3):
    torch.manual_seed(seed)
    net = LineNet().to(device).eval()
    rng = np.random.default_rng(seed)
    tables = [None, _sampled_table(rng), _sampled_table(rng)]
    checked = 0
    for t_i, table in enumerate(tables):
        env = _make_env(n_envs, rng, table)
        for _step in range(25):
            states = env.observe()
            x = encode_line_states(states, device, table)
            r_pred = net.immediate_rewards(x).cpu().numpy()
            valid = states[:, :25] == 0
            assert np.abs(r_pred[~valid]).max(initial=0.0) < atol, \
                "reward previsto non nullo su cella occupata"
            actions = tk.random_valid_actions(valid, rng)
            rewards = _step_env(env, actions)
            picked = r_pred[np.arange(n_envs), actions]
            err = np.abs(picked - rewards).max()
            assert err < atol, (
                f"decomposizione errata (tabella {t_i}, step {_step}): "
                f"max err {err:.4f}"
            )
            checked += n_envs
    print(f"[OK] 2/4 decomposizione reward: {checked} mosse verificate "
          f"(default + 2 tabelle campionate), err < {atol}")


def validate_d4_equivariance(seed=123, device="cpu", n_states=48, atol=2e-4):
    torch.manual_seed(seed + 1)
    net = LineNet().to(device).eval()
    rng = np.random.default_rng(seed + 1)
    perm, inv = _build_d4_perms()

    env = _make_env(n_states, rng)
    for _ in range(int(rng.integers(4, 18))):
        states = env.observe()
        valid = states[:, :25] == 0
        _step_env(env, tk.random_valid_actions(valid, rng))
    states = env.observe()

    for table in [None, _sampled_table(rng)]:
        with torch.no_grad():
            q = net(encode_line_states(states, device, table))
        for g in range(8):
            ts = states.copy()
            ts[:, :25] = states[:, perm[g]]
            with torch.no_grad():
                tq = net(encode_line_states(ts, device, table))
            gathered = tq[:, torch.from_numpy(inv[g]).to(tq.device)]
            err = (gathered - q).abs().max().item()
            assert err < atol, f"equivarianza D4 violata (g={g}): max err {err:.2e}"
    print(f"[OK] 3/4 equivarianza D4 esatta delle Q (8 simmetrie, 2 tabelle, "
          f"err < {atol})")


def validate_shapes_and_backward(seed=123, device="cpu"):
    torch.manual_seed(seed + 2)
    net = LineNet().to(device)
    rng = np.random.default_rng(seed + 2)

    empty = np.zeros((4, 26), dtype=np.uint8)
    empty[:, 25] = rng.integers(2, 13, size=4)

    env = _make_env(4, rng)
    for _ in range(11):
        states = env.observe()
        valid = states[:, :25] == 0
        _step_env(env, tk.random_valid_actions(valid, rng))
    mid = env.observe()

    terminal = np.zeros((4, 26), dtype=np.uint8)
    terminal[:, :25] = rng.integers(2, 13, size=(4, 25))
    terminal[:, 25] = 0

    for name, batch in [("iniziale", empty), ("intermedio", mid), ("terminale", terminal)]:
        q = net(encode_line_states(batch, device))
        assert q.shape == (4, 25), f"shape errata su stato {name}"
        assert torch.isfinite(q).all(), f"Q non finite su stato {name}"

    x = encode_line_states(mid, device)
    q = net(x)
    valid_t = torch.from_numpy((mid[:, :25] == 0)).to(device)
    loss = q[valid_t].mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "gradienti non finiti"
    assert any(g.abs().sum() > 0 for g in grads), "gradienti tutti nulli"
    print(f"[OK] 4/4 forma/finitezza + backward  |  parametri: {net.n_parameters:,}")


def run_all_validators(device="cpu", seed=123):
    print(f"LineNet — validatori (device={device}, seed={seed})")
    print(f"tabella default: {np.asarray(tk.DEFAULT_SCORE_TABLE).tolist()}  "
          f"| moltiplicatore diagonali: {DIAG_MULT:g}")
    validate_line_encoding(seed, device)
    validate_reward_decomposition(seed, device)
    validate_d4_equivariance(seed, device)
    validate_shapes_and_backward(seed, device)
    print("TUTTI I VALIDATORI SUPERATI")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validatori LineNet")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    run_all_validators(device=args.device, seed=args.seed)
