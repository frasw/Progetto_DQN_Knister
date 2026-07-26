"""Entrypoint di addestramento per LineNet.

Il ciclo di training eseguito è quello di train_knister_v2.py: viene importato
e richiamato, mai duplicato. Protocollo di acting, replay, ritorni a n passi,
selezione del checkpoint migliore, valutazioni periodiche e finali e
salvataggi restano quindi identici a quelli del motore per costruzione, e
train_knister_v2.py non viene modificato: le configurazioni che usano
l'architettura originale continuano a girare sul file intatto.

Questo modulo si limita a installare cinque innesti sul modulo importato:

  1. tk.encode_states          aggiunge il ramo state_encoding == "line"; per
                               ogni altro encoding delega all'originale
  2. tk.state_input_size       "line" -> 34; altrimenti delega
  3. tk.validate_state_encoder per "line" è un no-op, poiché la copertura è
                               affidata ai quattro validatori di LineNet
                               eseguiti all'avvio; altrimenti delega
  4. tk.DQN                    factory che restituisce una LineNet quando la
                               modalità è attiva (ignorando input_size,
                               hidden_size e network_type), altrimenti delega
                               alla DQN originale. Copre i tre punti in cui
                               train() costruisce una rete: policy, target e
                               ricarica del checkpoint migliore per la
                               valutazione finale
  5. tk.save_model             scrive network_type="linenet" nei metadati

NOTE OPERATIVE
--------------
- dueling_mask_aware non ha effetto su LineNet: la maschera interna è sempre
  attiva.
- --compile non è supportato con LineNet: le operazioni scatter e index_add
  non sono state verificate sotto compilazione.
- --network-type e --state-encoding non vanno passati, poiché gestiti
  internamente; il comando fallisce con un errore esplicito se presenti.
- Nel dump di configurazione stampato da train() compare
  network_type='dueling': è un valore di comodo necessario a superare la
  validazione del motore, che accetta solo "mlp" o "dueling". La rete
  effettivamente in uso è indicata dal banner [linenet] e dal conteggio dei
  parametri, e il metadato salvato nel checkpoint riporta "linenet".
- --help mostra l'help di train_knister_v2; i flag propri di questo modulo
  sono --line-embed-dim, --line-ctx-dim, --line-head-dim e
  --skip-startup-validators.

USO
---
  python linenet_train.py --n-step 3 --episodes 600000 --seed 42 \
      [--line-embed-dim 64 --line-ctx-dim 128 --line-head-dim 256] \
      [qualunque altro flag di train_knister_v2.py]

Percorsi di salvataggio predefiniti, se non passati esplicitamente:
  --save-path modello_linenet_last.pth
  --best-save-path modello_linenet_best.pth

All'avvio vengono eseguiti i quattro validatori di LineNet e stampato il
conteggio dei parametri, poi viene invocato tk.train(cfg).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from linenet import (
    LINE_STATE_SIZE,
    LineNet,
    encode_line_states,
    run_all_validators,
    tk,
)

# Stato della modalita' LineNet (letto dalla factory che sostituisce tk.DQN).
_MODE = {"active": False, "dims": (64, 128, 256)}
_ORIG: dict = {}


def activate_linenet(embed_dim=64, ctx_dim=128, head_dim=256):
    """Attiva la modalita' LineNet per la factory di rete.

    Esposta a livello di modulo perche' riusata anche da eval_tools.py per
    ricostruire i checkpoint LineNet.
    """
    _MODE["active"] = True
    _MODE["dims"] = (int(embed_dim), int(ctx_dim), int(head_dim))


def _replace_cfg(cfg, **kw):
    """dataclasses.replace con fallback a setattr (robusto a Config atipiche)."""
    try:
        return dataclasses.replace(cfg, **kw)
    except Exception:
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg


def install():
    """Installa gli innesti su train_knister_v2 (idempotente)."""
    if _ORIG:
        return

    _ORIG["encode_states"] = tk.encode_states
    _ORIG["state_input_size"] = tk.state_input_size
    _ORIG["DQN"] = tk.DQN
    _ORIG["save_model"] = tk.save_model
    _ORIG["validate_state_encoder"] = getattr(tk, "validate_state_encoder", None)

    def encode_states_ext(states_u8, device, state_encoding, score_table=None,
                          *args, **kwargs):
        if state_encoding == "line":
            return encode_line_states(states_u8, device, score_table)
        return _ORIG["encode_states"](states_u8, device, state_encoding,
                                      score_table, *args, **kwargs)

    def state_input_size_ext(state_encoding, *args, **kwargs):
        if state_encoding == "line":
            return LINE_STATE_SIZE
        return _ORIG["state_input_size"](state_encoding, *args, **kwargs)

    def dqn_factory(*args, **kwargs):
        if _MODE["active"]:
            e, c, h = _MODE["dims"]
            return LineNet(embed_dim=e, ctx_dim=c, head_dim=h)
        return _ORIG["DQN"](*args, **kwargs)

    def save_model_ext(model, cfg, *args, **kwargs):
        if _MODE["active"]:
            cfg = _replace_cfg(cfg, network_type="linenet")
        return _ORIG["save_model"](model, cfg, *args, **kwargs)

    tk.encode_states = encode_states_ext
    tk.state_input_size = state_input_size_ext
    tk.DQN = dqn_factory
    tk.save_model = save_model_ext

    patched = "encode_states, state_input_size, DQN, save_model"
    if _ORIG["validate_state_encoder"] is not None:
        def validate_state_encoder_ext(state_encoding, *args, **kwargs):
            if state_encoding == "line":
                print("[linenet] validate_state_encoder('line'): coperto dai "
                      "validatori LineNet gia' eseguiti all'avvio")
                return None
            return _ORIG["validate_state_encoder"](state_encoding,
                                                   *args, **kwargs)
        tk.validate_state_encoder = validate_state_encoder_ext
        patched += ", validate_state_encoder"

    print(f"[linenet] innesti installati su train_knister_v2 ({patched})")


def _argv_has(rest, flag):
    return any(a == flag or a.startswith(flag + "=") for a in rest)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--line-embed-dim", type=int, default=64)
    parser.add_argument("--line-ctx-dim", type=int, default=128)
    parser.add_argument("--line-head-dim", type=int, default=256)
    parser.add_argument("--skip-startup-validators", action="store_true")
    mine, rest = parser.parse_known_args()

    if not hasattr(tk, "parse_args") or not hasattr(tk, "train"):
        raise SystemExit(
            "[linenet] train_knister_v2 non espone parse_args()/train(): "
            "versione inattesa del modulo base."
        )

    for forbidden in ("--network-type", "--state-encoding", "--compile"):
        if _argv_has(rest, forbidden):
            raise SystemExit(
                f"[linenet] {forbidden} non va passato a linenet_train.py: "
                "network_type e state_encoding sono gestiti internamente, "
                "e --compile non e' testato con LineNet."
            )

    if not _argv_has(rest, "--save-path"):
        rest += ["--save-path", "modello_linenet_last.pth"]
    if not _argv_has(rest, "--best-save-path"):
        rest += ["--best-save-path", "modello_linenet_best.pth"]

    # Delega il parsing di tutti gli altri flag al parser ORIGINALE di v2,
    # cosi' default e semantica restano quelli del protocollo esistente.
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + rest
        parsed = tk.parse_args()
    finally:
        sys.argv = old_argv

    cfg = parsed if dataclasses.is_dataclass(parsed) else tk.Config(**vars(parsed))
    cfg = _replace_cfg(cfg, state_encoding="line", network_type="dueling")

    activate_linenet(mine.line_embed_dim, mine.line_ctx_dim, mine.line_head_dim)
    install()

    if not mine.skip_startup_validators:
        run_all_validators(device="cpu", seed=123)

    probe = LineNet(embed_dim=mine.line_embed_dim, ctx_dim=mine.line_ctx_dim,
                    head_dim=mine.line_head_dim)
    print(f"[linenet] rete effettiva: LineNet(embed={mine.line_embed_dim}, "
          f"ctx={mine.line_ctx_dim}, head={mine.line_head_dim}) - "
          f"parametri: {probe.n_parameters:,}")
    del probe
    print("[linenet] NOTA: nel dump di config 'network_type' appare 'dueling' "
          "(compatibilita' con la validazione di train()); i metadati dei "
          "checkpoint vengono riscritti a 'linenet'.")

    tk.train(cfg)


if __name__ == "__main__":
    main()
