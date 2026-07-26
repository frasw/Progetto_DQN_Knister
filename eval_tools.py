"""
eval_tools.py — PEZZO 3/3: strumenti di valutazione per checkpoint Knister.
================================================================================
Task (sottocomandi):

  info       carica un checkpoint, ricostruisce la rete, stampa metadati e
             una sanity greedy veloce (256 partite)
  lookahead  expectimax a 1 passo in valutazione: per ogni mossa
             argmax_a [ r(s,a) + Σ_v p(v) · max_a' Q(s',v,a') ] con la
             distribuzione ESATTA dei dadi (25×11 successori, forward batchato).
             Riporta greedy e lookahead sugli stessi semi (dadi appaiati) +
             percentuale di accordo tra le due politiche.
  zeroshot   stress test di generalizzazione: valuta greedy sotto la tabella
             default + N tabelle campionate, con feature "aware" (encoder
             alimentato con la NUOVA tabella) e "frozen" (encoder con la
             default mentre l'ambiente usa la nuova). Per checkpoint
             engineered192 aggiunge la modalita' "split" (proiezioni aware,
             blocco di condizionamento a 8 dim congelato alla default).
  official   verifica sul KnisterGame ufficiale via tk.evaluate_official,
             con adattamento per introspezione della firma (vedi nota sotto).

RICOSTRUZIONE DEL CHECKPOINT
----------------------------
1) si tenta la lettura dei metadati (dict/config nel payload o sidecar .json);
2) l'architettura viene comunque INFERITA dalle shape dello state_dict
   (fonte di verita': i pesi), con cross-check sui metadati:
   - chiavi phi.*/ctx_lin.* -> LineNet; dims: E=phi.0.weight[0],
     C=ctx_lin.weight[0], H=adv_head.0.weight[0]
   - altrimenti DQN v2: input_size/hidden_size dal primo peso 2-D;
     network_type dai metadati (default "dueling"); state_encoding dai
     metadati o dalla mappa input_size {184,192,259,267,312,26}.

NOTA DI TRASPARENZA (task `official`)
-------------------------------------
tk.evaluate_official e' l'unica funzione di v2 la cui firma non e' coperta dai
contratti gia' validati nei pezzi 1-2. Il task la invoca mappando i parametri
per introspezione (inspect.signature) da un dizionario di candidati; se la
mappatura non basta, stampa firma e docstring da riportare, senza toccare
nient'altro. Tutti gli altri task usano SOLO contratti gia' verificati:
step() = (next_state, reward, done), observe(), random_valid_actions,
score_lines_torch (via LineNet.immediate_rewards, validatore 2),
encode_line_states, encode_states (smoke del pezzo 2).

USO
---
  python eval_tools.py info      --checkpoint CKPT
  python eval_tools.py lookahead --checkpoint CKPT [--games 2000] [--seed 13579]
                                 [--batch-games 256] [--chunk 8192] [--device auto]
  python eval_tools.py zeroshot  --checkpoint CKPT [--games 2000] [--n-tables 8]
                                 [--table-seed 777] [--seed 13579]
                                 [--batch-games 512] [--device auto]
  python eval_tools.py official  --checkpoint CKPT [--games 500] [--seed 13579]

Il seme di valutazione di default (13579) e' DIVERSO dai semi di protocollo
9876/24681357 per non contaminare selezione BEST e hold-out.
================================================================================
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import sys

import numpy as np
import torch

import linenet_train as lt
from linenet import (
    LineNet,
    _make_env,
    _sampled_table,
    _step_env,
    encode_line_states,
    tk,
)

# ----------------------------------------------------------------------------
# util
# ----------------------------------------------------------------------------
_ENC_BY_INPUT = {184: ("engineered184", False), 192: ("engineered192", False),
                 259: ("engineered184", True), 267: ("engineered192", True),
                 312: ("onehot", False), 26: ("raw", False)}


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch datati senza il kwarg
        return torch.load(path, map_location="cpu")


def _is_tensor_dict(obj):
    return (isinstance(obj, dict) and obj
            and all(isinstance(v, torch.Tensor) for v in obj.values()))


def _strip_prefixes(sd):
    for pref in ("_orig_mod.", "module."):
        if all(k.startswith(pref) for k in sd):
            sd = {k[len(pref):]: v for k, v in sd.items()}
    return sd


def _extract_payload(path):
    """Ritorna (state_dict, metadati) da formati di checkpoint eterogenei."""
    payload = _torch_load(path)
    meta, sd = {}, None

    if _is_tensor_dict(payload):
        sd = payload
    elif isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights",
                    "policy_state_dict"):
            if _is_tensor_dict(payload.get(key)):
                sd = payload[key]
                break
        if sd is None:
            for v in payload.values():
                if _is_tensor_dict(v):
                    sd = v
                    break
        for v in list(payload.values()) + [payload]:
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                meta.update(dataclasses.asdict(v))
            elif isinstance(v, dict) and (
                    "state_encoding" in v or "network_type" in v):
                meta.update(v)
        for k, v in payload.items():
            if isinstance(v, (int, float, str, bool)):
                meta.setdefault(k, v)

    if sd is None:
        sd = tk.load_model_weights(path, torch.device("cpu"))
    sd = _strip_prefixes(dict(sd))

    for side in (path + ".json", os.path.splitext(path)[0] + ".json"):
        if os.path.exists(side):
            try:
                with open(side) as fh:
                    extra = json.load(fh)
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        meta.setdefault(k, v)
            except Exception as exc:  # sidecar malformato: non bloccante
                print(f"[eval_tools] sidecar {side} ignorato ({exc})")
    return sd, meta


def _call_with_matching_kwargs(fn, candidates):
    sig = inspect.signature(fn)
    kwargs, missing = {}, []
    for name, par in sig.parameters.items():
        if par.kind in (par.VAR_POSITIONAL, par.VAR_KEYWORD):
            continue
        if name in candidates:
            kwargs[name] = candidates[name]
        elif par.default is par.empty:
            missing.append(name)
    if missing:
        raise TypeError(f"parametri obbligatori senza candidato: {missing} "
                        f"(firma: {sig})")
    return fn(**kwargs)


def _stats(totals):
    totals = np.asarray(totals, dtype=np.float64)
    mean = totals.mean()
    std = totals.std(ddof=1) if totals.size > 1 else 0.0
    se = std / np.sqrt(totals.size) if totals.size else 0.0
    return mean, std, se


def _fmt(mean, std, se, n):
    return f"{mean:7.3f} ± {std:5.2f} (SE {se:.3f}, n={n})"


# ----------------------------------------------------------------------------
# Agent: checkpoint -> rete + encoder coerente
# ----------------------------------------------------------------------------
class Agent:
    def __init__(self, path, device):
        self.device = device
        sd, meta = _extract_payload(path)
        self.meta = meta

        if any(k.startswith("phi.") for k in sd) and "ctx_lin.weight" in sd:
            self.kind = "linenet"
            e = sd["phi.0.weight"].shape[0]
            c = sd["ctx_lin.weight"].shape[0]
            h = sd["adv_head.0.weight"].shape[0]
            self.dims = (int(e), int(c), int(h))
            net = LineNet(embed_dim=e, ctx_dim=c, head_dim=h)
            net.load_state_dict(sd, strict=True)
            self.state_encoding = "line"
            self.include_cls = False
            self.wide_norm = False
            self.input_size = 34
        else:
            self.kind = "mlp"
            first2d = next(t for t in sd.values() if t.dim() == 2)
            self.input_size = int(first2d.shape[1])
            hidden = int(first2d.shape[0])
            enc_guess, icls_guess = _ENC_BY_INPUT.get(
                self.input_size, ("engineered192", False))
            self.state_encoding = meta.get("state_encoding", enc_guess)
            self.include_cls = bool(meta.get("include_current_line_scores",
                                             icls_guess))
            self.wide_norm = bool(meta.get("randomize_scores", False))
            ntype = meta.get("network_type", "dueling")
            self.dims = (self.input_size, hidden, ntype)
            dqn_cls = lt._ORIG.get("DQN", tk.DQN)
            net = _call_with_matching_kwargs(dqn_cls, {
                "input_size": self.input_size, "hidden_size": hidden,
                "network_type": ntype, "output_size": 25, "n_actions": 25,
                "num_actions": 25,
            })
            try:
                net.load_state_dict(sd, strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    "load_state_dict fallito sulla DQN ricostruita "
                    f"(input={self.input_size}, hidden={hidden}, tipo={ntype}). "
                    f"Dettaglio: {exc}") from exc

        self.net = net.to(device).eval()
        self.n_parameters = sum(p.numel() for p in net.parameters())
        self._orig_encode = lt._ORIG.get("encode_states", tk.encode_states)
        self.mask_aware = bool(meta.get("dueling_mask_aware", True))
        self.pass_mask = True
        self._probe()

    # -- encoding -------------------------------------------------------------
    def _encode(self, states_u8, feat_table):
        if self.kind == "linenet":
            return encode_line_states(states_u8, self.device, feat_table)
        sig = inspect.signature(self._orig_encode)
        kwargs = {}
        if "wide_norm" in sig.parameters:
            kwargs["wide_norm"] = self.wide_norm
        if "include_current_line_scores" in sig.parameters:
            kwargs["include_current_line_scores"] = self.include_cls
        return self._orig_encode(states_u8, self.device, self.state_encoding,
                                 feat_table, **kwargs)

    @property
    def split_supported(self):
        return self.kind == "mlp" and self.input_size == 192

    @torch.no_grad()
    def q_values(self, states_u8, feat_table=None, cond_frozen_split=False):
        x = self._encode(states_u8, feat_table)
        if cond_frozen_split:
            if not self.split_supported:
                raise ValueError("modalita' split solo per engineered192")
            x_def = self._encode(states_u8, None)
            x = x.clone()
            x[:, -8:] = x_def[:, -8:]
        mask = None
        if self.pass_mask:
            valid = states_u8[:, :25] == 0
            mask = torch.from_numpy(np.ascontiguousarray(valid)).to(self.device)
            if not self.mask_aware and self.kind == "mlp":
                mask = None
        try:
            return self.net(x, mask) if self.pass_mask else self.net(x)
        except TypeError:
            self.pass_mask = False
            return self.net(x)

    def _probe(self):
        rng = np.random.default_rng(0)
        env = _make_env(2, rng)
        _step_env(env, tk.random_valid_actions(env.observe()[:, :25] == 0, rng))
        q = self.q_values(env.observe())
        assert q.shape == (2, 25) and torch.isfinite(q).all(), \
            "probe forward fallita sul checkpoint caricato"

    def describe(self):
        if self.kind == "linenet":
            e, c, h = self.dims
            arch = f"LineNet(embed={e}, ctx={c}, head={h})"
        else:
            i, hdim, nt = self.dims
            arch = f"DQN v2 (tipo={nt}, input={i}, hidden={hdim})"
        return (f"{arch} | parametri: {self.n_parameters:,} | "
                f"encoding: {self.state_encoding} | "
                f"cls={self.include_cls} wide_norm={self.wide_norm} "
                f"mask_aware={self.mask_aware}")


# ----------------------------------------------------------------------------
# valutazioni
# ----------------------------------------------------------------------------
def evaluate_greedy(agent, games, seed, env_table=None, feat_table=None,
                    cond_frozen_split=False, batch_games=512):
    rng = np.random.default_rng(seed)
    device = agent.device
    totals = []
    remaining = games
    while remaining > 0:
        b = min(batch_games, remaining)
        env = _make_env(b, rng, env_table)
        score = np.zeros(b, dtype=np.float32)
        for _t in range(25):
            states = env.observe()
            q = agent.q_values(states, feat_table, cond_frozen_split)
            valid = torch.from_numpy(
                np.ascontiguousarray(states[:, :25] == 0)).to(device)
            q = q.masked_fill(~valid, torch.finfo(q.dtype).min)
            actions = q.argmax(dim=1).cpu().numpy().astype(np.int64)
            score += _step_env(env, actions)
        totals.append(score)
        remaining -= b
    return np.concatenate(totals)


class _Lookahead:
    """Expectimax a 1 passo con modello dei dadi esatto e Q come foglia."""

    def __init__(self, agent, chunk=8192):
        self.agent = agent
        self.chunk = int(chunk)
        probs = np.asarray(tk.DICE_SUM_PROBABILITIES, dtype=np.float32)
        assert probs.shape == (11,), "attese 11 probabilita' di somma dadi"
        self.probs = probs
        # rete usa-e-getta: immediate_rewards e' strutturale (indipendente
        # dai pesi, validatore 2), dims minime per costo trascurabile
        self._imm = LineNet(embed_dim=8, ctx_dim=8, head_dim=8)
        self._imm = self._imm.to(agent.device).eval()
        self.moves = 0
        self.agree = 0

    def actions(self, states, t, feat_table=None):
        agent, device = self.agent, self.agent.device
        b = states.shape[0]
        grid = states[:, :25].astype(np.int64)
        valid = grid == 0

        x_line = encode_line_states(states, device, feat_table)
        r_all = self._imm.immediate_rewards(x_line).cpu().numpy()   # [b, 25]

        q_now = agent.q_values(states, feat_table)
        q_now = q_now.masked_fill(
            ~torch.from_numpy(valid).to(device), torch.finfo(q_now.dtype).min)
        greedy_a = q_now.argmax(dim=1).cpu().numpy()

        if t == 24:
            total = np.where(valid, r_all, -np.inf)
        else:
            ar = np.arange(25)
            succ = np.repeat(states[:, None, :], 25, axis=1)        # [b,25,26]
            succ[:, ar, ar] = states[:, 25][:, None]                # piazza il lancio
            succ11 = np.repeat(succ[:, :, None, :], 11, axis=2)     # [b,25,11,26]
            succ11[..., 25] = np.arange(2, 13, dtype=np.uint8)[None, None, :]
            flat = np.ascontiguousarray(succ11.reshape(b * 275, 26))

            sv = np.repeat(valid[:, None, :], 25, axis=1)           # [b,25,25]
            sv[:, ar, ar] = False
            sv_flat = sv[:, :, None, :].repeat(11, axis=2).reshape(b * 275, 25)

            maxq = np.empty(b * 275, dtype=np.float32)
            for start in range(0, b * 275, self.chunk):
                sl = slice(start, min(start + self.chunk, b * 275))
                q = agent.q_values(flat[sl], feat_table)
                vm = torch.from_numpy(
                    np.ascontiguousarray(sv_flat[sl])).to(device)
                q = q.masked_fill(~vm, torch.finfo(q.dtype).min)
                maxq[sl] = q.max(dim=1).values.float().cpu().numpy()

            ev = (maxq.reshape(b, 25, 11) * self.probs[None, None, :]).sum(axis=2)
            total = np.where(valid, r_all + ev, -np.inf)

        actions = total.argmax(axis=1).astype(np.int64)
        self.moves += b
        self.agree += int((actions == greedy_a).sum())
        return actions


def evaluate_lookahead(agent, games, seed, chunk=8192, batch_games=256,
                       env_table=None, feat_table=None):
    rng = np.random.default_rng(seed)
    la = _Lookahead(agent, chunk=chunk)
    totals = []
    remaining = games
    while remaining > 0:
        b = min(batch_games, remaining)
        env = _make_env(b, rng, env_table)
        score = np.zeros(b, dtype=np.float32)
        for t in range(25):
            states = env.observe()
            actions = la.actions(states, t, feat_table)
            score += _step_env(env, actions)
        totals.append(score)
        remaining -= b
    return np.concatenate(totals), la.agree / max(la.moves, 1)


# ----------------------------------------------------------------------------
# task
# ----------------------------------------------------------------------------
def task_info(args, agent):
    print(agent.describe())
    if agent.meta:
        keys = ("network_type", "state_encoding", "n_step", "episodes",
                "hidden_size", "seed", "eval_mean", "eval_std",
                "checkpoint_type", "episodes_done")
        shown = {k: agent.meta[k] for k in keys if k in agent.meta}
        print(f"metadati (estratto): {shown}")
    totals = evaluate_greedy(agent, 256, seed=999,
                             batch_games=args.batch_games)
    m, s, se = _stats(totals)
    print(f"sanity greedy 256 partite (seed 999): {_fmt(m, s, se, totals.size)}")


def task_lookahead(args, agent):
    print(f"greedy baseline: {args.games} partite, seed {args.seed} ...")
    g = evaluate_greedy(agent, args.games, args.seed,
                        batch_games=args.batch_games)
    gm, gs, gse = _stats(g)
    print(f"  greedy    : {_fmt(gm, gs, gse, g.size)}")

    print(f"lookahead expectimax 1-passo (25x11 successori/mossa) ...")
    l, agree = evaluate_lookahead(agent, args.games, args.seed,
                                  chunk=args.chunk,
                                  batch_games=args.batch_games)
    lm, ls, lse = _stats(l)
    print(f"  lookahead : {_fmt(lm, ls, lse, l.size)}")
    dse = float(np.sqrt(gse ** 2 + lse ** 2))
    print(f"RISULTATO lookahead: Δ = {lm - gm:+.3f} (SE diff ≈ {dse:.3f}, "
          f"dadi appaiati per seme) | accordo azioni greedy/lookahead: "
          f"{agree * 100:.1f}%")


def task_zeroshot(args, agent):
    rng = np.random.default_rng(args.table_seed)
    names = list(getattr(tk, "SCORE_TABLE_NAMES",
                         [f"c{i}" for i in range(8)]))
    default = np.asarray(tk.DEFAULT_SCORE_TABLE, dtype=np.float32)
    tables = [("default", None)]
    for i in range(args.n_tables):
        tables.append((f"T{i + 1}", _sampled_table(rng)))

    modes = ["aware", "frozen"] + (["split"] if agent.split_supported else [])
    print(f"tabella default: {dict(zip(names, default.tolist()))}")
    print(f"modalita': {modes}  (aware = encoder con la nuova tabella; "
          f"frozen = encoder con la default; split = solo proiezioni aware)")
    baseline = None
    for tname, table in tables:
        if table is not None:
            print(f"\n{tname}: {dict(zip(names, np.asarray(table).tolist()))}")
        else:
            print(f"\n{tname}")
        for mode in modes:
            if table is None and mode != "aware":
                continue  # sulla default le modalita' coincidono
            feat = table if mode in ("aware", "split") else None
            totals = evaluate_greedy(
                agent, args.games, args.seed, env_table=table,
                feat_table=feat, cond_frozen_split=(mode == "split"),
                batch_games=args.batch_games)
            m, s, se = _stats(totals)
            if baseline is None:
                baseline = m
            print(f"  {mode:6s}: {_fmt(m, s, se, totals.size)}   "
                  f"Δ vs default = {m - baseline:+7.3f}")
    print("\nRISULTATO zeroshot: righe sopra (media sotto ciascuna tabella, "
          "Δ rispetto alla default).")


def task_official(args, agent):
    fn = getattr(tk, "evaluate_official", None)
    if fn is None:
        print("[eval_tools] tk.evaluate_official non trovato: mandami "
              "l'elenco delle funzioni di valutazione del tuo v2.")
        return
    # cfg minimale coi default del parser originale, coerente col checkpoint
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        parsed = tk.parse_args()
    finally:
        sys.argv = old_argv
    cfg = parsed if dataclasses.is_dataclass(parsed) else tk.Config(**vars(parsed))
    cfg = lt._replace_cfg(cfg, state_encoding=agent.state_encoding,
                          include_current_line_scores=agent.include_cls,
                          randomize_scores=agent.wide_norm)

    candidates = {
        "model": agent.net, "policy": agent.net, "net": agent.net,
        "policy_net": agent.net, "device": agent.device,
        "n_games": args.games, "num_games": args.games, "games": args.games,
        "games_count": args.games,
        "n_episodes": args.games, "episodes": args.games,
        "seed": args.seed, "eval_seed": args.seed,
        "state_encoding": agent.state_encoding,
        "include_current_line_scores": agent.include_cls,
        "wide_norm": agent.wide_norm, "score_table": None,
        "batch_size": args.batch_games, "eval_batch_size": args.batch_games,
        "cfg": cfg, "config": cfg,
    }
    if agent.kind == "linenet":
        lt.install()  # il dispatch 'line' serve dentro evaluate_official
    print(f"verifica su API ufficiale: {args.games} partite, seed {args.seed} ...")
    try:
        result = _call_with_matching_kwargs(fn, candidates)
        print(f"RISULTATO official: {result}")
    except Exception as exc:
        print(f"[eval_tools] adattamento della firma fallito: {exc}")
        try:
            print(f"firma: {inspect.signature(fn)}")
            print(f"doc: {inspect.getdoc(fn)}")
        except Exception:
            pass
        print("Mandami firma/doc qui sopra: adatto la chiamata in una riga.")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Strumenti di valutazione Knister")
    sub = parser.add_subparsers(dest="task", required=True)
    for name in ("info", "lookahead", "zeroshot", "official"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--device", default="auto")
        p.add_argument("--seed", type=int, default=13579)
        p.add_argument("--games", type=int,
                       default=500 if name == "official" else 2000)
        p.add_argument("--batch-games", type=int,
                       default=256 if name == "lookahead" else 512)
        if name == "lookahead":
            p.add_argument("--chunk", type=int, default=8192)
        if name == "zeroshot":
            p.add_argument("--n-tables", type=int, default=8)
            p.add_argument("--table-seed", type=int, default=777)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    lt.install()  # dispatch 'line' disponibile ovunque; innocuo per gli mlp
    agent = Agent(args.checkpoint, device)
    print(f"[eval_tools] checkpoint: {args.checkpoint}")
    print(f"[eval_tools] {agent.describe()} | device: {device}")

    {"info": task_info, "lookahead": task_lookahead,
     "zeroshot": task_zeroshot, "official": task_official}[args.task](args, agent)


if __name__ == "__main__":
    main()
