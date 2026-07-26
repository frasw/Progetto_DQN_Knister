"""Knister Double-DQN: ambiente vettorizzato, training e valutazione.

Questo modulo contiene il motore completo del progetto: la reimplementazione
vettorizzata del gioco, gli encoder di stato, la rete DQN, il replay buffer,
il ciclo di addestramento e le procedure di valutazione.

CONFORMITA' ALLE REGOLE DEL GIOCO
---------------------------------
I punteggi delle combinazioni non sono costanti cablate nel codice: la tabella
di default viene letta una sola volta dalle costanti pubbliche di
api.KnisterGame (DEFAULT_SCORE_TABLE) e da lì passata come parametro esplicito
a ogni funzione che ne ha bisogno (FastVectorKnister._score_lines,
score_lines_torch e gli encoder). Una modifica dei punteggi si propaga quindi
in modo coerente, senza copie che restano disallineate.

L'equivalenza tra l'ambiente vettorizzato e api.KnisterGame è verificata da
validate_fast_environment(), eseguita all'avvio di ogni addestramento: le
stesse partite vengono giocate con entrambe le implementazioni e se ne
confrontano reward, griglie e punteggi finali.

ARCHITETTURA DELLA RETE
-----------------------
DQN supporta due varianti, selezionabili con Config.network_type: un MLP
semplice e una versione dueling (V(s) + A(s,a) - media(A)). Nella variante
dueling la media delle advantage è calcolata sulle sole azioni valide, non su
tutte e 25: le celle già occupate non sono scelte legali e non devono ricevere
contributo di gradiente attraverso il termine di media. Il comportamento è
controllato da Config.dueling_mask_aware (default True); impostarlo a False
riproduce la media non mascherata, utile solo per confronti diretti.

OPZIONI DI ADDESTRAMENTO
------------------------
Il modulo espone diverse varianti algoritmiche, tutte disattivate di default e
combinabili liberamente. Questo consente di riprodurre dallo stesso file sia la
configurazione finale sia gli esperimenti di confronto.

1. RITORNI A N PASSI (Config.n_step, default 1). Con gamma=1 il ritorno a n
   passi è la somma dei reward reali nella finestra, senza pesi esponenziali,
   troncata correttamente in prossimità della fine dell'episodio (l'orizzonte
   è sempre fisso a 25 passi). Richiede di bufferizzare l'intera traiettoria
   di un round prima di inserire le transizioni nel replay: gli aggiornamenti
   di gradiente avvengono quindi in blocco dopo ogni round completo
   (25 * updates_per_vector_step) invece che intrecciati passo per passo. A
   parità di episodi il numero totale di aggiornamenti resta lo stesso.

2. TABELLA PUNTEGGI VARIABILE (Config.randomize_scores, default False).
   Campiona una tabella diversa a ogni round vettoriale durante il training.
   Va usata con l'encoder "engineered192", che include la tabella corrente tra
   le feature: la policy impara così a condizionarsi sui punteggi invece di
   assumerli fissi. La valutazione usa sempre la tabella reale, indipendentemente
   da questa opzione, così il punteggio riportato resta confrontabile.

3. SCORE CORRENTE PER LINEA (Config.include_current_line_scores, default
   False). Aggiunge 75 feature allineate alle azioni con il punteggio corrente
   (non proiettato) di riga, colonna e diagonale di ciascuna azione, riusando
   un calcolo già presente nell'encoder per derivare il reward immediato.
   Componibile con gli altri encoder: 184+75=259 dimensioni con
   "engineered184", 192+75=267 con "engineered192".

4. PRIORITIZED EXPERIENCE REPLAY (Config.prioritized_replay, default False).
   Campionamento proporzionale a (|TD-error| + per_epsilon) ** per_alpha, con
   correzione di importance sampling nella loss e beta annealed linearmente da
   per_beta_start a per_beta_end. Si appoggia a SumTree, un albero delle
   priorità che rende campionamento e aggiornamento O(log n) invece di O(n).
   Con l'opzione disattivata il campionamento è uniforme; ReplayBuffer.sample
   restituisce comunque indici e pesi (sempre 1.0), così train_one_batch ha un
   unico percorso di codice.

5. SIMMETRIE D4 COME DATA AUGMENTATION (Config.symmetry_augmentation, default
   False). A ogni minibatch campionato dal replay applica una delle 8
   trasformazioni del gruppo diedrale (identità, 3 rotazioni, 4 riflessioni) a
   stato, stato successivo e azione, scelta indipendentemente per ciascun
   campione. I reward non vanno trasformati: il punteggio Knister dipende solo
   dal multiset di valori di ogni linea, non da quale riga o colonna fisica
   sia. L'invarianza è verificata da validate_symmetry_augmentation(),
   eseguita all'avvio quando l'opzione è attiva. L'augmentation avviene al
   momento del campionamento e non aumenta la memoria del replay buffer.

6. LEARNING RATE DECRESCENTE (Config.lr_end, default None = costante). Se
   impostato, il learning rate resta fisso fino alla frazione
   lr_decay_start_fraction degli aggiornamenti stimati, poi decresce
   linearmente fino a lr_end alla frazione lr_decay_fraction, restando poi
   costante. Il valore corrente compare nel log di progresso solo quando il
   decadimento è attivo. Nota sperimentale: un decadimento avviato fin
   dall'inizio del training si è rivelato dannoso, perché riduce il passo
   proprio nella fase di crescita più rapida; lr_decay_start_fraction esiste
   per limitare il decadimento al tratto finale.

ENCODER DI STATO
----------------
state_encoding seleziona la rappresentazione passata alla rete: "raw" (26
dimensioni), "onehot" (312), "engineered184" e "engineered192". Le varianti
engineered comprendono valori normalizzati delle celle, maschera di validità,
reward immediato e proiezioni di linea per ciascuna azione, lancio corrente in
one-hot, progresso della partita e istogrammi dei valori presenti e attesi.
La variante 192 aggiunge la tabella dei punteggi corrente.

VALUTAZIONE
-----------
Due modalità: "vectorized" usa FastVectorKnister ed è quella impiegata per le
valutazioni periodiche durante il training; "official" gioca direttamente con
api.KnisterGame ed è più lenta, pensata per la verifica finale di un
checkpoint. Entrambe usano una politica greedy pura, senza esplorazione.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timedelta
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from api import KnisterGame


RAW_STATE_SIZE = 26
N_CELLS = 25
GRID_SIZE = 5

# -----------------------------------------------------------------------------
# Simmetrie D4 della griglia 5x5, per data augmentation nel replay.
# Il punteggio Knister dipende solo dal
# multiset di valori presente in ciascuna riga/colonna/diagonale, non da quale
# riga/colonna fisica sia: applicare una delle 8 trasformazioni del gruppo
# diedrale a griglia+azione produce quindi una transizione ALTRETTANTO VALIDA
# (stesso reward, stesso score finale), non una approssimazione. Verificato
# empiricamente in validate_symmetry_augmentation().
_INDEX_GRID = np.arange(N_CELLS, dtype=np.int64).reshape(GRID_SIZE, GRID_SIZE)
_D4_GRIDS = [
    _INDEX_GRID,                        # identità
    np.rot90(_INDEX_GRID, k=1),         # rotazione 90°
    np.rot90(_INDEX_GRID, k=2),         # rotazione 180°
    np.rot90(_INDEX_GRID, k=3),         # rotazione 270°
    np.fliplr(_INDEX_GRID),             # flip orizzontale
    np.flipud(_INDEX_GRID),             # flip verticale
    _INDEX_GRID.T,                      # trasposta (flip diagonale principale)
    _INDEX_GRID[::-1, ::-1].T,          # flip diagonale anti (rot180 poi trasposta)
]
N_D4 = len(_D4_GRIDS)
# D4_FORWARD_PERM[g, k] = indice, nella griglia originale, del valore che
# finisce in posizione k dopo la trasformazione g (semantica "gather").
D4_FORWARD_PERM = np.stack([g.flatten().copy() for g in _D4_GRIDS]).astype(np.int64)
# D4_INVERSE_PERM[g, a] = nuova posizione della cella che, nella griglia
# originale, era in posizione a — serve per trasformare l'azione.
D4_INVERSE_PERM = np.stack([np.argsort(p) for p in D4_FORWARD_PERM]).astype(np.int64)


def apply_symmetry_to_states(states_u8: np.ndarray, group_idx: np.ndarray) -> np.ndarray:
    """Applica, riga per riga, la trasformazione D4 group_idx[i] (0..7) alla
    griglia (prime 25 colonne). Il lancio (colonna 25) è invariante."""
    perm_rows = D4_FORWARD_PERM[group_idx]  # [batch, 25]
    out = states_u8.copy()
    out[:, :N_CELLS] = np.take_along_axis(states_u8[:, :N_CELLS], perm_rows, axis=1)
    return out


def apply_symmetry_to_actions(actions: np.ndarray, group_idx: np.ndarray) -> np.ndarray:
    inv_perm_rows = D4_INVERSE_PERM[group_idx]  # [batch, 25]
    return np.take_along_axis(inv_perm_rows, actions[:, None].astype(np.int64), axis=1)[:, 0]

GRID_CATEGORIES = 12  # vuoto + valori 2..12
ROLL_CATEGORIES = 11  # valori 2..12
ONEHOT_STATE_SIZE = N_CELLS * GRID_CATEGORIES + ROLL_CATEGORIES + 1  # 312
ENGINEERED184_STATE_SIZE = 184
ENGINEERED192_STATE_SIZE = 184 + 8  # 184 feature originali + tabella punteggi corrente
# Score CORRENTE (non proiettato) di riga/colonna/diagonale, allineato alle 25
# azioni come i blocchi "projected" esistenti (25+25+25). Richiesto da un
# "cos'è presente nelle righe/colonne/diagonali" (es. "riga 2 tris")
# — l'informazione era già calcolata internamente per derivare il reward
# immediato, ma veniva scartata invece di essere esposta come feature.
CURRENT_LINE_SCORES_SIZE = 75

# Probabilità delle somme di due dadi, per i valori 2..12.
DICE_SUM_PROBABILITIES = np.array(
    [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=np.float32
) / 36.0

# -----------------------------------------------------------------------------
# Tabella di scoring: unica fonte di verità = api.KnisterGame.
#
# In precedenza i punteggi (10, 6, 8, 3, 3, 1, 8, 12) erano duplicati come
# letterali sia in FastVectorKnister._score_lines sia in score_lines_torch,
# senza alcun collegamento a api.py. Qui la tabella di default
# viene letta una sola volta dalle costanti pubbliche di KnisterGame, e da quel
# momento è sempre un parametro esplicito passato a chi ne ha bisogno, mai un
# letterale sparso nel codice.
#
# Ordine fisso e canonico usato ovunque nel file:
#   0=cinquina, 1=poker, 2=full house, 3=doppia coppia,
#   4=tris, 5=coppia, 6=scala con 7, 7=scala senza 7
# -----------------------------------------------------------------------------

SCORE_TABLE_SIZE = 8
SCORE_TABLE_NAMES = (
    "cinquina", "poker", "full_house", "doppia_coppia",
    "tris", "coppia", "scala_con_7", "scala_senza_7",
)
DEFAULT_SCORE_TABLE = np.array(
    [
        KnisterGame.FIVE_OF_A_KIND,
        KnisterGame.FOUR_OF_A_KIND,
        KnisterGame.FULL_HOUSE,
        KnisterGame.TWO_PAIRS,
        KnisterGame.THREE_OF_A_KIND,
        KnisterGame.ONE_PAIR,
        KnisterGame.STRAIGHT_WITH_SEVEN,
        KnisterGame.STRAIGHT_NO_SEVEN,
    ],
    dtype=np.int32,
)
# Il moltiplicatore diagonale non è, per come è formulato il compito ("i
# punteggi associati alle diverse combo"), una combo: resta fisso, ma letto
# comunque dalla fonte ufficiale invece che riscritto a mano.
DIAGONAL_MULTIPLIER = float(KnisterGame.DIAGONAL_MULTIPLIER)

# Range di jitter moltiplicativo usato dalla domain randomization (vedi
# sample_score_table). I limiti sono prudenziali: coprono variazioni
# ragionevoli della tabella dei punteggi senza assumerne una specifica.
SCORE_JITTER_LOW = 0.5
SCORE_JITTER_HIGH = 2.0

# Massimali usati per normalizzare le feature dipendenti dallo score
# nell'encoder. Due varianti:
#   TIGHT: i valori esatti della versione precedente (12, 48, 72) — validi
#          solo se la tabella è sempre quella di default. Usati quando
#          randomize_scores=False, per riprodurre bit-per-bit le feature di
#          prima (compatibilità con la scala su cui la rete precedente si
#          era assestata).
#   WIDE:  massimali allargati al range di jitter configurato — necessari
#          quando randomize_scores=True, altrimenti le feature normalizzate
#          potrebbero superare 1.0 sotto una tabella scalata verso l'alto.
def _score_ceilings(jitter_high: float) -> tuple[float, float, float]:
    line = float(DEFAULT_SCORE_TABLE.max()) * jitter_high
    diag = 2.0 * DIAGONAL_MULTIPLIER * line
    immediate = 2.0 * line + diag
    return line, diag, immediate


LINE_SCORE_CEILING_TIGHT, DIAG_SCORE_CEILING_TIGHT, IMMEDIATE_REWARD_CEILING_TIGHT = (
    _score_ceilings(1.0)
)
LINE_SCORE_CEILING, DIAG_SCORE_CEILING, IMMEDIATE_REWARD_CEILING = _score_ceilings(
    SCORE_JITTER_HIGH
)


def sample_score_table(
    rng: np.random.Generator,
    base: np.ndarray = DEFAULT_SCORE_TABLE,
    jitter_low: float = SCORE_JITTER_LOW,
    jitter_high: float = SCORE_JITTER_HIGH,
) -> np.ndarray:
    """Perturba ogni punteggio della tabella base con un fattore indipendente
    in [jitter_low, jitter_high], arrotonda a intero, minimo 1.

    Ogni combo viene scalata da un fattore casuale diverso (non un fattore
    unico per tutta la tabella): l'ordine relativo tra combo può quindi
    cambiare, il che è intenzionale — costringe la rete a leggere davvero la
    tabella corrente invece di affidarsi a un ordinamento memorizzato.
    """
    factors = rng.uniform(jitter_low, jitter_high, size=SCORE_TABLE_SIZE)
    table = np.round(base.astype(np.float64) * factors).astype(np.int32)
    return np.maximum(table, 1)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class Config:
    episodes: int = 3_000_000
    n_envs: int = 1024
    replay_capacity: int = 2_000_000
    batch_size: int = 1024
    learning_starts: int = 100_000
    updates_per_vector_step: int = 2

    gamma: float = 1.0
    lr: float = 2.5e-4
    # None = lr costante per tutto il training (comportamento precedente,
    # identico bit per bit). Se impostato: lr resta fisso a `lr` fino alla
    # frazione `lr_decay_start_fraction` degli aggiornamenti gradiente totali
    # stimati, poi decresce linearmente fino a `lr_end` alla frazione
    # `lr_decay_fraction`, poi resta costante a lr_end. Con
    # lr_decay_start_fraction=0.0 (default) il decadimento parte subito —
    # ATTENZIONE, notare nel changelog: su un run lungo questo può tagliare
    # il lr proprio nella fase di crescita più forte. Per un vero "tapering"
    # finale serve alzare lr_decay_start_fraction (es. 0.8).
    lr_end: Optional[float] = None
    lr_decay_start_fraction: float = 0.0
    lr_decay_fraction: float = 1.0
    hidden_size: int = 384
    network_type: str = "dueling"
    # "engineered192" = "engineered184" + 8 feature con la tabella di scoring
    # corrente (normalizzata): necessario per condizionare la policy sui
    # punteggi quando randomize_scores=True. "engineered184" resta disponibile
    # come baseline/controllo per l'ablation. "onehot" e "raw" invariati.
    state_encoding: str = "engineered192"
    grad_clip: float = 10.0
    target_update_every: int = 4_000  # gradient updates, non passi ambiente

    # --- Conformità: scoring configurabile ---
    # Se False: ogni round usa DEFAULT_SCORE_TABLE, comportamento identico
    # alla versione precedente (utile come controllo/ablation).
    randomize_scores: bool = False
    score_jitter_low: float = SCORE_JITTER_LOW
    score_jitter_high: float = SCORE_JITTER_HIGH

    # --- Dueling mask-aware ---
    # True: la media delle advantage nella testa dueling è calcolata solo
    # sulle azioni valide. False: riproduce esattamente il comportamento
    # precedente (media su tutte le 25), utile solo per confronti diretti
    # tra le due varianti.
    dueling_mask_aware: bool = True

    # --- N-step return ---
    # 1 = transizioni a singolo step, comportamento identico alla versione
    # precedente. n>1 = ritorno a n-step (esatto per gamma=1, l'orizzonte è
    # sempre 25 e fisso quindi non serve troncare per episodi variabili).
    n_step: int = 1

    # --- Score corrente di riga/colonna/diagonale ---
    # +75 feature action-aligned, riuso di un calcolo già presente nell'encoder
    # (prima scartato dopo aver derivato il reward immediato). Vedi
    # encode_engineered per il layout esatto.
    include_current_line_scores: bool = False

    # --- Prioritized Experience Replay ---
    prioritized_replay: bool = False
    per_alpha: float = 0.6          # 0 = uniforme, 1 = priorità piena (Schaul et al.)
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_epsilon: float = 1e-3       # evita priorità nulla per TD-error=0

    # --- Simmetrie D4 come data augmentation nel replay.
    # Trasformazione casuale indipendente per
    # ciascun campione del minibatch, applicata al momento del sampling: non
    # aumenta la memoria del replay buffer.
    symmetry_augmentation: bool = False

    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay_fraction: float = 0.50

    seed: int = 42
    device: str = "auto"
    compile_model: bool = False
    cpu_threads: int = 4

    # Stampa un report ogni circa X episodi. Poiché gli ambienti vengono
    # eseguiti in blocchi, il numero effettivo può superare leggermente la soglia.
    log_every_episodes: int = 20_000
    progress_window: int = 5
    log_every_batches: int = 20
    eval_every_episodes: int = 100_000
    eval_games: int = 500
    eval_seed: int = 9_876
    # Test finale separato, eseguito sul checkpoint BEST con un seed diverso.
    final_eval_games: int = 10_000
    final_eval_seed: int = 24_681_357
    # "vectorized" usa FastVectorKnister ed è la modalità consigliata.
    # "official" usa api.KnisterGame in blocchi e mostra il progresso ogni N match.
    eval_mode: str = "vectorized"
    eval_batch_size: int = 1024
    eval_progress_every: int = 50
    save_path: str = "modello_knister_v2_last.pth"
    best_save_path: str = "modello_knister_v2_best.pth"


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------


class DQN(nn.Module):
    def __init__(
        self,
        input_size: int = ENGINEERED184_STATE_SIZE,
        output_size: int = 25,
        hidden_size: int = 384,
        network_type: str = "dueling",
    ):
        super().__init__()
        if network_type not in {"mlp", "dueling"}:
            raise ValueError("network_type deve essere 'mlp' oppure 'dueling'")
        self.network_type = network_type
        self.output_size = output_size
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        if network_type == "dueling":
            head_size = max(64, hidden_size // 2)
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, head_size),
                nn.ReLU(),
                nn.Linear(head_size, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(hidden_size, head_size),
                nn.ReLU(),
                nn.Linear(head_size, output_size),
            )
        else:
            self.q_head = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """valid_mask: bool tensor [batch, output_size], True = azione disponibile.

        Se fornita e network_type=="dueling", la media delle advantage usata
        per centrare Q è calcolata solo sulle azioni valide:
        senza questa maschera, l'advantage delle celle occupate riceve comunque
        un contributo di gradiente attraverso il termine di media, anche se
        quella cella non è mai selezionabile in quello stato). Per network_type
        "mlp" l'argomento è ignorato: non c'è aggregazione da correggere.
        """
        features = self.trunk(x)
        if self.network_type == "dueling":
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            if valid_mask is not None:
                valid_f = valid_mask.to(dtype=advantage.dtype)
                valid_count = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                adv_mean = (advantage * valid_f).sum(dim=1, keepdim=True) / valid_count
            else:
                adv_mean = advantage.mean(dim=1, keepdim=True)
            return value + advantage - adv_mean
        return self.q_head(features)


# -----------------------------------------------------------------------------
# Fast vectorized environment, semantically equivalent to api.KnisterGame
# -----------------------------------------------------------------------------


class FastVectorKnister:
    """Esegue molte partite di Knister sincronizzate in NumPy.

    La classe mantiene la semantica ufficiale delle ricompense: la ricompensa è
    la differenza tra il punteggio totale dopo e prima del posizionamento.
    Vengono ricalcolati i punteggi solo per la riga, la colonna e (eventualmente) 
    le diagonali interessate.

    """

    SIZE = 5
    N_CELLS = 25
    N_LINES = 12  # 5 righe + 5 colonne + 2 diagonali

    def __init__(
        self,
        n_envs: int,
        rng: np.random.Generator,
        score_table: Optional[np.ndarray] = None,
    ):
        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        self.n_envs = int(n_envs)
        self.rng = rng
        self._env_ids = np.arange(self.n_envs)
        self._one_hot = np.eye(13, dtype=np.uint8)
        self.grids = np.zeros((self.n_envs, self.N_CELLS), dtype=np.uint8)
        self.rolls = np.zeros(self.n_envs, dtype=np.uint8)
        # int32 (non più int16): sotto tabelle di scoring randomizzate i punteggi
        # possono superare i massimi originali; int32 lascia ampio margine.
        self.line_scores = np.zeros((self.n_envs, self.N_LINES), dtype=np.int32)
        self.total_scores = np.zeros(self.n_envs, dtype=np.int32)
        self.step_index = 0
        self.score_table = (
            DEFAULT_SCORE_TABLE.copy()
            if score_table is None
            else np.asarray(score_table, dtype=np.int32)
        )
        if self.score_table.shape != (SCORE_TABLE_SIZE,):
            raise ValueError(
                f"score_table deve avere shape ({SCORE_TABLE_SIZE},), "
                f"ricevuta {self.score_table.shape}"
            )
        self.reset()

    def _sample_rolls(self) -> np.ndarray:
        # Somma di due dadi a sei facce indipendenti e non truccati.
        return (
            self.rng.integers(1, 7, size=self.n_envs, dtype=np.uint8)
            + self.rng.integers(1, 7, size=self.n_envs, dtype=np.uint8)
        )

    def reset(self, first_rolls: Optional[np.ndarray] = None) -> np.ndarray:
        self.grids.fill(0)
        self.line_scores.fill(0)
        self.total_scores.fill(0)
        self.step_index = 0
        if first_rolls is None:
            self.rolls[:] = self._sample_rolls()
        else:
            first_rolls = np.asarray(first_rolls, dtype=np.uint8)
            if first_rolls.shape != (self.n_envs,):
                raise ValueError("first_rolls has wrong shape")
            self.rolls[:] = first_rolls
        return self.observe()

    def observe(self) -> np.ndarray:
        states = np.empty((self.n_envs, 26), dtype=np.uint8)
        states[:, :25] = self.grids
        states[:, 25] = self.rolls
        return states

    def valid_action_mask(self) -> np.ndarray:
        return self.grids == 0

    def _score_lines(self, lines: np.ndarray) -> np.ndarray:
        """Equivalente vettorializzato di KnisterGame.score_line per la shape [N, 5]."""
        lines = np.asarray(lines, dtype=np.uint8)
        filled = np.count_nonzero(lines, axis=1)

        # Conteggi per i valori 2..12. I valori 0 e 1 non contribuiscono.
        counts = self._one_hot[lines].sum(axis=1)[:, 2:13]
        sorted_counts = np.sort(counts, axis=1)[:, ::-1]
        first = sorted_counts[:, 0]
        second = sorted_counts[:, 1]

        st = self.score_table
        scores = np.zeros(lines.shape[0], dtype=np.int32)
        scores[first == 5] = st[0]
        scores[first == 4] = st[1]
        scores[(first == 3) & (second == 2)] = st[2]
        scores[(first == 2) & (second == 2)] = st[3]
        scores[(first == 3) & (second != 2)] = st[4]
        scores[(first == 2) & (second != 2)] = st[5]

        # Una scala è possibile solo con cinque valori distinti e consecutivi.
        full_unique = (filled == 5) & (first == 1)
        min_value = lines.min(axis=1)
        max_value = lines.max(axis=1)
        straight = full_unique & ((max_value - min_value) == 4)
        contains_seven = np.any(lines == 7, axis=1)
        scores[straight & contains_seven] = st[6]
        scores[straight & ~contains_seven] = st[7]
        return scores

    def step(
        self,
        actions: np.ndarray,
        forced_next_rolls: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.step_index >= self.N_CELLS:
            raise RuntimeError("Tutte le partite vettorializzate sono già terminate")

        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n_envs,):
            raise ValueError("le azioni hanno una shape sbagliata")
        if np.any((actions < 0) | (actions >= self.N_CELLS)):
            raise ValueError("azioni fuori dal range [0, 24]")
        if np.any(self.grids[self._env_ids, actions] != 0):
            raise ValueError("tentativo di eseguire un'azione non valida o su una posizione occupata")

        old_totals = self.total_scores.copy()
        self.grids[self._env_ids, actions] = self.rolls

        grid_3d = self.grids.reshape(self.n_envs, self.SIZE, self.SIZE)
        rows = actions // self.SIZE
        cols = actions % self.SIZE

        row_values = grid_3d[self._env_ids, rows, :]
        col_values = grid_3d[self._env_ids, :, cols]
        self.line_scores[self._env_ids, rows] = self._score_lines(row_values)
        self.line_scores[self._env_ids, self.SIZE + cols] = self._score_lines(col_values)

        on_main_diag = rows == cols
        if np.any(on_main_diag):
            ids = self._env_ids[on_main_diag]
            subgrid = grid_3d[ids]
            diag = subgrid[:, np.arange(self.SIZE), np.arange(self.SIZE)]
            self.line_scores[ids, 10] = DIAGONAL_MULTIPLIER * self._score_lines(diag)

        on_anti_diag = (rows + cols) == (self.SIZE - 1)
        if np.any(on_anti_diag):
            ids = self._env_ids[on_anti_diag]
            subgrid = grid_3d[ids]
            diag = subgrid[:, np.arange(self.SIZE), np.arange(self.SIZE - 1, -1, -1)]
            self.line_scores[ids, 11] = DIAGONAL_MULTIPLIER * self._score_lines(diag)

        self.total_scores[:] = self.line_scores.sum(axis=1, dtype=np.int32)
        rewards = (self.total_scores - old_totals).astype(np.int32, copy=False)

        self.step_index += 1
        done = np.full(self.n_envs, self.step_index == self.N_CELLS, dtype=np.bool_)

        if done[0]:
            self.rolls.fill(0)
        elif forced_next_rolls is None:
            self.rolls[:] = self._sample_rolls()
        else:
            forced_next_rolls = np.asarray(forced_next_rolls, dtype=np.uint8)
            if forced_next_rolls.shape != (self.n_envs,):
                raise ValueError("forced_next_rolls ha una shape errata")
            self.rolls[:] = forced_next_rolls

        return self.observe(), rewards, done


# -----------------------------------------------------------------------------
# Replay buffer memorizzato in modo compatto nella RAM della CPU
# -----------------------------------------------------------------------------


class SumTree:
    """Albero binario completo su array: le foglie sono le priorità delle
    transizioni, ogni nodo interno è la somma dei due figli. Permette
    campionamento proporzionale alla priorità e aggiornamento delle priorità
    in O(log capacity), invece di O(capacity) per un ricalcolo ingenuo della
    distribuzione cumulativa ad ogni sample. Implementazione standard per
    Prioritized Experience Replay.

    `data_capacity` non deve essere una potenza di 2: internamente si usa la
    prima potenza di 2 >= data_capacity (`tree_capacity`); le foglie in più
    restano a priorità 0 e non vengono mai campionate.
    """

    def __init__(self, data_capacity: int):
        if data_capacity <= 0:
            raise ValueError("data_capacity deve essere positivo")
        self.data_capacity = int(data_capacity)
        tree_capacity = 1
        while tree_capacity < self.data_capacity:
            tree_capacity *= 2
        self.tree_capacity = tree_capacity
        self.depth = int(round(math.log2(tree_capacity)))
        self.tree = np.zeros(2 * tree_capacity, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[1])

    def max_leaf(self) -> float:
        leaves = self.tree[self.tree_capacity : self.tree_capacity + self.data_capacity]
        m = float(leaves.max()) if leaves.size else 0.0
        return m if m > 0 else 1.0

    def update_batch(self, data_indices: np.ndarray, priorities: np.ndarray) -> None:
        """Imposta la priorità di più foglie in una singola passata vettorizzata.

        Se lo stesso data_index compare più volte in questa chiamata, si tiene
        solo l'ULTIMA priorità assegnata (stessa semantica "ultimo vince" con
        cui viene comunque sovrascritta la foglia), evitando di sommare più
        volte lo stesso delta durante la propagazione verso la radice.
        """
        data_indices = np.asarray(data_indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)
        if data_indices.shape != priorities.shape:
            raise ValueError("data_indices e priorities devono avere la stessa shape")
        if data_indices.size == 0:
            return

        if np.unique(data_indices).size != data_indices.size:
            order = np.arange(data_indices.size)
            sort_key = np.lexsort((order, data_indices))
            sorted_idx = data_indices[sort_key]
            sorted_prio = priorities[sort_key]
            is_last = np.empty(sorted_idx.size, dtype=bool)
            is_last[:-1] = sorted_idx[:-1] != sorted_idx[1:]
            is_last[-1] = True
            data_indices = sorted_idx[is_last]
            priorities = sorted_prio[is_last]

        leaf_indices = data_indices + self.tree_capacity
        deltas = priorities - self.tree[leaf_indices]
        self.tree[leaf_indices] = priorities

        node_indices = leaf_indices.copy()
        for _ in range(self.depth):
            node_indices = node_indices // 2
            np.add.at(self.tree, node_indices, deltas)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Campionamento stratificato: divide [0, total) in batch_size
        segmenti uguali e pesca un punto uniforme in ciascuno (riduce la
        varianza rispetto al campionamento i.i.d. puro). 
        Interamente vettorizzato: self.depth iterazioni totali,
        non batch_size * self.depth.
        """
        total = self.total()
        if total <= 0:
            raise RuntimeError("SumTree vuoto: nessuna priorità positiva da campionare")
        segment = total / batch_size
        los = segment * np.arange(batch_size, dtype=np.float64)
        cumulative = los + rng.uniform(0.0, segment, size=batch_size)
        cumulative = np.minimum(cumulative, total - 1e-9)

        node_indices = np.ones(batch_size, dtype=np.int64)
        for _ in range(self.depth):
            left = 2 * node_indices
            left_val = self.tree[left]
            go_left = cumulative <= left_val
            cumulative = np.where(go_left, cumulative, cumulative - left_val)
            node_indices = np.where(go_left, left, left + 1)

        data_indices = node_indices - self.tree_capacity
        priorities = self.tree[node_indices]
        return data_indices, priorities


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self.states = np.empty((capacity, RAW_STATE_SIZE), dtype=np.uint8)
        self.next_states = np.empty((capacity, RAW_STATE_SIZE), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.uint8)
        self.rewards = np.empty(capacity, dtype=np.int32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        # Tabella di scoring in vigore quando la transizione è stata raccolta.
        # Serve perché, con randomize_scores=True, un minibatch campionato dal
        # replay può mescolare transizioni di round diversi con tabelle diverse.
        # Senza questo campo, train_one_batch userebbe
        # per errore la tabella "corrente" anche per transizioni vecchie.
        # Con randomize_scores=False è sempre DEFAULT_SCORE_TABLE: costo
        # trascurabile (32 byte/transizione) mantenuto per semplicità, invece
        # di un campo opzionale.
        self.score_tables = np.empty((capacity, SCORE_TABLE_SIZE), dtype=np.int32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        score_tables: np.ndarray,
    ) -> np.ndarray:
        """Ritorna gli indici del buffer effettivamente scritti (servono a
        PrioritizedReplayBuffer per inizializzare le priorità delle nuove
        transizioni)."""
        n = states.shape[0]
        if score_tables.shape != (n, SCORE_TABLE_SIZE):
            raise ValueError(
                f"score_tables deve avere shape ({n}, {SCORE_TABLE_SIZE}), "
                f"ricevuta {score_tables.shape}"
            )
        if n > self.capacity:
            states = states[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_states = next_states[-self.capacity :]
            dones = dones[-self.capacity :]
            score_tables = score_tables[-self.capacity :]
            n = self.capacity

        first = min(n, self.capacity - self.pos)
        second = n - first
        sl = slice(self.pos, self.pos + first)
        self.states[sl] = states[:first]
        self.actions[sl] = actions[:first]
        self.rewards[sl] = rewards[:first]
        self.next_states[sl] = next_states[:first]
        self.dones[sl] = dones[:first]
        self.score_tables[sl] = score_tables[:first]

        if second:
            sl2 = slice(0, second)
            self.states[sl2] = states[first:]
            self.actions[sl2] = actions[first:]
            self.rewards[sl2] = rewards[first:]
            self.next_states[sl2] = next_states[first:]
            self.dones[sl2] = dones[first:]
            self.score_tables[sl2] = score_tables[first:]

        written = (self.pos + np.arange(n)) % self.capacity
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        return written

    def sample(self, batch_size: int, beta: float = 1.0) -> tuple[np.ndarray, ...]:
        """`beta` è ignorato qui (nessuna correzione da fare: campionamento
        già uniforme); presente solo per condividere l'interfaccia con
        PrioritizedReplayBuffer.sample, così train_one_batch non deve sapere
        quale dei due buffer sta usando.
        """
        idx = self.rng.integers(0, self.size, size=batch_size)
        is_weights = np.ones(batch_size, dtype=np.float32)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.score_tables[idx],
            idx,
            is_weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """No-op: il buffer uniforme non ha priorità da aggiornare. Firma
        identica a PrioritizedReplayBuffer.update_priorities."""
        return


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay: le transizioni con
    TD-error assoluto maggiore vengono campionate più spesso, con un peso di
    importance sampling (IS) che corregge il bias introdotto nella loss.
    Stesso layout dati di ReplayBuffer.
    (stati compatti uint8 in RAM); in più un SumTree con le priorità.
    """

    def __init__(
        self,
        capacity: int,
        rng: np.random.Generator,
        alpha: float = 0.6,
        epsilon: float = 1e-3,
    ):
        super().__init__(capacity, rng)
        self.tree = SumTree(capacity)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.max_priority = 1.0  # priorità assegnata alle transizioni ancora mai valutate

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        score_tables: np.ndarray,
    ) -> np.ndarray:
        written = super().add_batch(states, actions, rewards, next_states, dones, score_tables)
        # Priorità massima corrente: garantisce che le transizioni appena
        # raccolte vengano campionate presto almeno una volta.
        init_priority = self.max_priority ** self.alpha
        self.tree.update_batch(written, np.full(written.shape[0], init_priority))
        return written

    def sample(self, batch_size: int, beta: float = 1.0) -> tuple[np.ndarray, ...]:
        data_indices, leaf_priorities = self.tree.sample(batch_size, self.rng)
        total = self.tree.total()
        probs = leaf_priorities / total
        # IS weight standard: (N * P(i))^-beta, normalizzato per il massimo
        # cosicché il peso più alto nel batch sia sempre 1 (stabilizza la
        # scala della loss, pratica comune nelle implementazioni di PER).
        weights = (self.size * probs) ** (-beta)
        weights = weights / weights.max()

        return (
            self.states[data_indices],
            self.actions[data_indices],
            self.rewards[data_indices],
            self.next_states[data_indices],
            self.dones[data_indices],
            self.score_tables[data_indices],
            data_indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.tree.update_batch(indices, priorities)


# -----------------------------------------------------------------------------
# Funzioni di supporto per DQN e codifica dello stato
# -----------------------------------------------------------------------------


def state_input_size(state_encoding: str, include_current_line_scores: bool = False) -> int:
    if state_encoding == "engineered192":
        base = ENGINEERED192_STATE_SIZE
    elif state_encoding == "engineered184":
        base = ENGINEERED184_STATE_SIZE
    elif state_encoding == "onehot":
        return ONEHOT_STATE_SIZE
    elif state_encoding == "raw":
        return RAW_STATE_SIZE
    else:
        raise ValueError(
            f"state_encoding non valido: {state_encoding!r}. "
            "Usare 'engineered192', 'engineered184', 'onehot' oppure 'raw'."
        )
    return base + (CURRENT_LINE_SCORES_SIZE if include_current_line_scores else 0)


def _broadcastable_score_columns(
    score_table: torch.Tensor, target: torch.Tensor
) -> list[torch.Tensor]:
    """Riporta score_table a 8 colonne broadcastabili contro `target`.

    score_table.dim()==1 (shape [8]): stessa tabella per tutto il batch — usato
    in fase di ACTING, dove un intero round vettoriale condivide una sola
    tabella (vedi Config.randomize_scores). Ogni colonna è uno scalare, che
    torch.where broadcasta automaticamente.

    score_table.dim()==2 (shape [batch, 8]): una tabella diversa per ciascun
    elemento del batch — necessario in fase di TRAINING, perché un minibatch
    campionato dal replay può contenere transizioni raccolte in round diversi,
    ciascuno con una tabella diversa. `target` può avere dimensioni intermedie
    aggiuntive rispetto al batch (es. le 25 azioni candidate in
    encode_engineered): qui si aggiungono le dimensioni singleton necessarie
    perché il broadcasting le allinei correttamente.
    """
    if score_table.dim() == 1:
        return [score_table[k] for k in range(SCORE_TABLE_SIZE)]
    extra_dims = target.dim() - 1  # dimensioni di target oltre al batch
    shape = (score_table.shape[0],) + (1,) * extra_dims
    return [score_table[:, k].reshape(shape) for k in range(SCORE_TABLE_SIZE)]


def score_lines_torch(lines: torch.Tensor, score_table: torch.Tensor) -> torch.Tensor:
    """Versione Torch vettorizzata e differenziabile solo rispetto all'input numerico.

    `lines` può avere qualunque shape iniziale, purché l'ultima dimensione sia 5.
    `score_table`: shape [8] (condivisa da tutto il batch) oppure [batch, 8]
    (una tabella per elemento del batch) — vedi _broadcastable_score_columns.
    Restituisce lo score Knister della linea come float32.
    """
    if lines.shape[-1] != GRID_SIZE:
        raise ValueError(f"L'ultima dimensione deve essere {GRID_SIZE}, ricevuta {lines.shape}")
    if score_table.shape[-1] != SCORE_TABLE_SIZE or score_table.dim() not in (1, 2):
        raise ValueError(
            f"score_table deve avere shape ({SCORE_TABLE_SIZE},) oppure "
            f"(batch, {SCORE_TABLE_SIZE}), ricevuta {tuple(score_table.shape)}"
        )

    lines = lines.to(dtype=torch.long)
    filled = (lines != 0).sum(dim=-1)
    counts = F.one_hot(lines.clamp(0, 12), num_classes=13)[..., 2:13].sum(dim=-2)
    sorted_counts = counts.sort(dim=-1, descending=True).values
    first = sorted_counts[..., 0]
    second = sorted_counts[..., 1]

    st = _broadcastable_score_columns(score_table.to(dtype=torch.float32), first)

    scores = torch.zeros_like(first, dtype=torch.float32)
    scores = torch.where(first == 5, st[0], scores)
    scores = torch.where(first == 4, st[1], scores)
    scores = torch.where((first == 3) & (second == 2), st[2], scores)
    scores = torch.where((first == 2) & (second == 2), st[3], scores)
    scores = torch.where((first == 3) & (second != 2), st[4], scores)
    scores = torch.where((first == 2) & (second != 2), st[5], scores)

    full_unique = (filled == 5) & (first == 1)
    min_value = lines.min(dim=-1).values
    max_value = lines.max(dim=-1).values
    straight = full_unique & ((max_value - min_value) == 4)
    contains_seven = (lines == 7).any(dim=-1)
    scores = torch.where(straight & contains_seven, st[6], scores)
    scores = torch.where(straight & ~contains_seven, st[7], scores)
    return scores


def encode_engineered(
    raw: torch.Tensor,
    score_table: torch.Tensor,
    condition_on_score: bool,
    wide_norm: bool = False,
    include_current_line_scores: bool = False,
) -> torch.Tensor:
    """Crea le 184 feature dense originali, più blocchi opzionali:
      - condition_on_score=True: +8 feature con la tabella di scoring corrente
        (184+8=192, modalità "engineered192").
      - include_current_line_scores=True: +75 feature con lo score CORRENTE
        (non proiettato) di riga/colonna/diagonale per ciascuna delle 25
        azioni — stessa informazione già calcolata internamente per derivare
        il reward immediato (righe 75:150 sono "dopo l'azione"; questo nuovo
        blocco è "adesso, prima dell'azione"), prima scartata dopo il calcolo
        del delta. Costo aggiuntivo pressoché nullo poichè i tensori esistono già.

    Layout (184 feature base, invariato rispetto alla versione precedente):
      0:25    valori griglia normalizzati
      25:50   maschera celle libere
      50:75   reward immediato previsto per ciascuna azione
      75:100  score della riga dopo ciascuna azione
      100:125 score della colonna dopo ciascuna azione
      125:150 contributo diagonale dopo ciascuna azione
      150:161 one-hot del dado corrente
      161     avanzamento del turno
      162:173 conteggio corrente dei valori 2..12
      173:184 conteggio finale atteso, dato ciò che resta da lanciare
    Blocchi opzionali, in quest'ordine se entrambi attivi:
      184:184+75           score corrente di riga/colonna/diagonale (se
                            include_current_line_scores=True)
      +0:+8 (dopo i sopra)  tabella di scoring corrente (se condition_on_score)

    `score_table` è sempre richiesto (non solo quando condition_on_score=True):
    serve comunque per calcolare correttamente reward/score proiettati, che
    devono riflettere la tabella realmente in uso in quel round — altrimenti,
    sotto randomize_scores=True, le feature 50:150 mentirebbero alla rete.

    Le feature 50:150 sono allineate alle 25 azioni: il neurone Q dell'azione i
    riceve quindi, attraverso una mappa fissa, informazioni già pertinenti alla
    cella i senza dover ricostruire da zero righe, colonne e diagonali. Questo è
    un aiuto alla rappresentazione, non un vincolo architetturale: una MLP fully
    connected non garantisce che l'output i "ascolti" davvero soprattutto il
    blocco i.
    """
    grid = raw[:, :N_CELLS].to(dtype=torch.long)
    roll = raw[:, N_CELLS].to(dtype=torch.long)
    n = grid.shape[0]
    device = grid.device
    score_table = score_table.to(device=device)

    valid = grid == 0
    valid_f = valid.to(dtype=torch.float32)
    grid_f = grid.to(dtype=torch.float32)
    grid_norm = grid_f * (1.0 / 12.0)

    action_ids = torch.arange(N_CELLS, device=device)
    row_ids = action_ids // GRID_SIZE
    col_ids = action_ids % GRID_SIZE

    # Una griglia candidata per ciascuna delle 25 azioni. Per le celle occupate
    # la griglia resta invariata e le relative feature vengono poi azzerate.
    candidates = grid.unsqueeze(1).expand(n, N_CELLS, N_CELLS).clone()
    roll_per_action = roll.unsqueeze(1).expand(-1, N_CELLS)
    placed_values = torch.where(valid, roll_per_action, grid)
    candidates[:, action_ids, action_ids] = placed_values
    candidates_4d = candidates.reshape(n, N_CELLS, GRID_SIZE, GRID_SIZE)

    row_gather = row_ids.view(1, N_CELLS, 1, 1).expand(n, -1, 1, GRID_SIZE)
    col_gather = col_ids.view(1, N_CELLS, 1, 1).expand(n, -1, GRID_SIZE, 1)
    candidate_rows = candidates_4d.gather(2, row_gather).squeeze(2)
    candidate_cols = candidates_4d.gather(3, col_gather).squeeze(3)

    projected_row_raw = score_lines_torch(candidate_rows, score_table)
    projected_col_raw = score_lines_torch(candidate_cols, score_table)

    grid_3d = grid.reshape(n, GRID_SIZE, GRID_SIZE)
    current_row_all = score_lines_torch(grid_3d, score_table)
    current_col_all = score_lines_torch(grid_3d.transpose(1, 2), score_table)
    current_row = current_row_all[:, row_ids]
    current_col = current_col_all[:, col_ids]

    candidate_main = candidates_4d.diagonal(dim1=2, dim2=3)
    candidate_anti = torch.flip(candidates_4d, dims=[3]).diagonal(dim1=2, dim2=3)
    projected_main = score_lines_torch(candidate_main, score_table)
    projected_anti = score_lines_torch(candidate_anti, score_table)

    current_main = score_lines_torch(grid_3d.diagonal(dim1=1, dim2=2), score_table)
    current_anti = score_lines_torch(
        torch.flip(grid_3d, dims=[2]).diagonal(dim1=1, dim2=2), score_table
    )
    main_mask = (row_ids == col_ids).to(dtype=torch.float32).unsqueeze(0)
    anti_mask = ((row_ids + col_ids) == (GRID_SIZE - 1)).to(dtype=torch.float32).unsqueeze(0)
    diag_mult = DIAGONAL_MULTIPLIER
    projected_diag_raw = diag_mult * projected_main * main_mask + diag_mult * projected_anti * anti_mask
    current_diag = diag_mult * current_main.unsqueeze(1) * main_mask + diag_mult * current_anti.unsqueeze(1) * anti_mask

    immediate_reward = (
        projected_row_raw - current_row
        + projected_col_raw - current_col
        + projected_diag_raw - current_diag
    ) * valid_f

    # wide_norm=False (default): massimali TIGHT, identici bit-per-bit alla
    # versione precedente (12/48/72) — corretti perché con randomize_scores=
    # False la tabella è sempre quella di default. wide_norm=True: massimali
    # WIDE, necessari quando la tabella può eccedere i valori originali.
    if wide_norm:
        line_ceiling, diag_ceiling, immediate_ceiling = (
            LINE_SCORE_CEILING, DIAG_SCORE_CEILING, IMMEDIATE_REWARD_CEILING
        )
    else:
        line_ceiling, diag_ceiling, immediate_ceiling = (
            LINE_SCORE_CEILING_TIGHT, DIAG_SCORE_CEILING_TIGHT, IMMEDIATE_REWARD_CEILING_TIGHT
        )
    immediate_reward_norm = immediate_reward * (1.0 / immediate_ceiling)
    projected_row_norm = projected_row_raw * valid_f * (1.0 / line_ceiling)
    projected_col_norm = projected_col_raw * valid_f * (1.0 / line_ceiling)
    projected_diag_norm = projected_diag_raw * valid_f * (1.0 / diag_ceiling)
    # Score CORRENTE (non mascherato da valid_f: ha senso anche per celle già
    # occupate, che appartengono comunque a una riga/colonna/diagonale con un
    # punteggio attuale ben definito) — vedi include_current_line_scores.
    current_row_norm = current_row * (1.0 / line_ceiling)
    current_col_norm = current_col * (1.0 / line_ceiling)
    current_diag_norm = current_diag * (1.0 / diag_ceiling)

    valid_roll = (roll >= 2) & (roll <= 12)
    roll_indices = roll.clamp(2, 12) - 2
    roll_onehot = F.one_hot(roll_indices, num_classes=ROLL_CATEGORIES).to(torch.float32)
    roll_onehot.mul_(valid_roll.unsqueeze(1))

    turn_count = (grid != 0).sum(dim=1, keepdim=True).to(dtype=torch.float32)
    turn_norm = turn_count * (1.0 / N_CELLS)

    board_counts = F.one_hot(grid.clamp(0, 12), num_classes=13)[..., 2:13].sum(dim=1)
    board_counts = board_counts.to(dtype=torch.float32)
    board_counts_norm = board_counts * (1.0 / N_CELLS)

    remaining = (N_CELLS - turn_count).clamp_min(0.0)
    probabilities = torch.as_tensor(DICE_SUM_PROBABILITIES, device=device).unsqueeze(0)
    forecast_counts_norm = (board_counts + remaining * probabilities) * (1.0 / N_CELLS)

    parts = [
        grid_norm,
        valid_f,
        immediate_reward_norm,
        projected_row_norm,
        projected_col_norm,
        projected_diag_norm,
        roll_onehot,
        turn_norm,
        board_counts_norm,
        forecast_counts_norm,
    ]
    expected_size = ENGINEERED184_STATE_SIZE

    if include_current_line_scores:
        parts.extend([current_row_norm, current_col_norm, current_diag_norm])
        expected_size += CURRENT_LINE_SCORES_SIZE

    if condition_on_score:
        st_f = score_table.to(dtype=torch.float32)
        # [8] (round condiviso, fase di acting) -> espande a [n,8].
        # [batch,8] (fase di training, un campione dal replay per riga) -> già
        # nella forma giusta, nessun expand necessario.
        score_norm = (st_f / line_ceiling) if st_f.dim() == 2 else (st_f / line_ceiling).unsqueeze(0).expand(n, -1)
        parts.append(score_norm)
        expected_size += SCORE_TABLE_SIZE

    encoded = torch.cat(parts, dim=1)
    if encoded.shape[1] != expected_size:
        raise RuntimeError(
            f"Encoder engineered ha prodotto {encoded.shape[1]} feature invece di "
            f"{expected_size}"
        )
    return encoded


def encode_states(
    states_u8: np.ndarray,
    device: torch.device,
    state_encoding: str,
    score_table: Optional[torch.Tensor] = None,
    wide_norm: bool = False,
    include_current_line_scores: bool = False,
) -> torch.Tensor:
    """Codifica sul device mantenendo il replay buffer compatto (26 byte/stato).

    `score_table`: usato solo da "engineered184"/"engineered192". Se omesso si
    usa DEFAULT_SCORE_TABLE (comportamento identico alla versione precedente).
    Va passato esplicitamente quando randomize_scores=True, con la tabella
    effettivamente in uso nel round corrente.
    `wide_norm`: True quando randomize_scores=True (massimali di normalizzazione
    allargati al range di jitter). Con False, "engineered184" riproduce
    esattamente la scala di feature della versione precedente.
    `include_current_line_scores`: aggiunge il blocco +75 con lo score
    corrente di riga/colonna/diagonale (vedi encode_engineered).
    """
    states_u8 = np.asarray(states_u8, dtype=np.uint8)
    if states_u8.ndim != 2 or states_u8.shape[1] != RAW_STATE_SIZE:
        raise ValueError(
            f"states_u8 deve avere shape [N, {RAW_STATE_SIZE}], ricevuta {states_u8.shape}"
        )

    raw = torch.from_numpy(states_u8).to(device=device, non_blocking=True)

    if state_encoding == "raw":
        return raw.to(dtype=torch.float32).mul_(1.0 / 12.0)
    if state_encoding in ("engineered184", "engineered192"):
        table = (
            torch.from_numpy(DEFAULT_SCORE_TABLE)
            if score_table is None
            else score_table
        ).to(device=device)
        return encode_engineered(
            raw,
            table,
            condition_on_score=(state_encoding == "engineered192"),
            wide_norm=wide_norm,
            include_current_line_scores=include_current_line_scores,
        )
    if state_encoding != "onehot":
        raise ValueError(
            f"state_encoding non valido: {state_encoding!r}. "
            "Usare 'engineered192', 'engineered184', 'onehot' oppure 'raw'."
        )

    grid = raw[:, :N_CELLS].to(dtype=torch.long)
    grid_indices = torch.where(grid == 0, torch.zeros_like(grid), grid - 1)
    grid_onehot = F.one_hot(
        grid_indices, num_classes=GRID_CATEGORIES
    ).reshape(grid.shape[0], -1).to(dtype=torch.float32)

    roll = raw[:, N_CELLS].to(dtype=torch.long)
    valid_roll = (roll >= 2) & (roll <= 12)
    roll_indices = roll.clamp(2, 12) - 2
    roll_onehot = F.one_hot(
        roll_indices, num_classes=ROLL_CATEGORIES
    ).to(dtype=torch.float32)
    roll_onehot.mul_(valid_roll.unsqueeze(1))

    turn = (grid != 0).sum(dim=1, keepdim=True).to(dtype=torch.float32)
    turn.mul_(1.0 / N_CELLS)

    encoded = torch.cat((grid_onehot, roll_onehot, turn), dim=1)
    if encoded.shape[1] != ONEHOT_STATE_SIZE:
        raise RuntimeError(
            f"Encoder one-hot ha prodotto {encoded.shape[1]} feature invece di "
            f"{ONEHOT_STATE_SIZE}"
        )
    return encoded


def validate_state_encoder(
    state_encoding: str, include_current_line_scores: bool = False
) -> None:
    """Sanity check di dimensione, finitezza e semantica dell'encoder."""
    sample = np.zeros((3, RAW_STATE_SIZE), dtype=np.uint8)
    sample[0, 25] = 2
    sample[1, :3] = np.array([2, 7, 12], dtype=np.uint8)
    sample[1, 25] = 9
    sample[2, :25] = 7
    sample[2, 25] = 0

    encoded = encode_states(
        sample, torch.device("cpu"), state_encoding,
        include_current_line_scores=include_current_line_scores,
    )
    expected_size = state_input_size(state_encoding, include_current_line_scores)
    if tuple(encoded.shape) != (3, expected_size):
        raise AssertionError(
            f"Shape encoder errata: {tuple(encoded.shape)}, attesa (3, {expected_size})"
        )
    if not torch.isfinite(encoded).all():
        raise AssertionError("L'encoder ha prodotto valori non finiti")

    if state_encoding == "onehot":
        grid_part = encoded[:, : N_CELLS * GRID_CATEGORIES]
        grid_sums = grid_part.reshape(3, N_CELLS, GRID_CATEGORIES).sum(dim=2)
        if not torch.allclose(grid_sums, torch.ones_like(grid_sums)):
            raise AssertionError("Ogni cella deve avere esattamente una categoria one-hot")
        roll_part = encoded[0, N_CELLS * GRID_CATEGORIES : -1]
        if not torch.isclose(roll_part.sum(), torch.tensor(1.0)):
            raise AssertionError("Il dado non terminale deve avere una categoria one-hot")
        terminal_roll = encoded[2, N_CELLS * GRID_CATEGORIES : -1]
        if not torch.isclose(terminal_roll.sum(), torch.tensor(0.0)):
            raise AssertionError("Il dado terminale deve essere codificato come vettore nullo")
        expected_turns = torch.tensor([0.0, 3.0 / 25.0, 1.0])
        if not torch.allclose(encoded[:, -1], expected_turns, atol=1e-6):
            raise AssertionError("Feature del turno normalizzato errata")

    if state_encoding in ("engineered184", "engineered192"):
        wide = state_encoding == "engineered192"
        ceiling = IMMEDIATE_REWARD_CEILING if wide else IMMEDIATE_REWARD_CEILING_TIGHT
        line_ceiling = LINE_SCORE_CEILING if wide else LINE_SCORE_CEILING_TIGHT
        current_block_offset = ENGINEERED184_STATE_SIZE  # 184, prima di eventuali blocchi extra
        # Verifica la feature del reward immediato sotto DUE tabelle: quella di
        # default e una randomizzata. Se cambiasse solo la prima e non la
        # seconda, un bug nel filo score_table -> ambiente/encoder passerebbe
        # inosservato con la sola tabella di default.
        for table in (DEFAULT_SCORE_TABLE, sample_score_table(np.random.default_rng(99))):
            table_t = torch.from_numpy(table.astype(np.float32))
            rng = np.random.default_rng(20_260_710)
            env = FastVectorKnister(32, rng, score_table=table)
            states = env.observe()
            for _ in range(7):
                actions = random_valid_actions(env.valid_action_mask(), rng)
                states, _, _ = env.step(actions)
            actions = random_valid_actions(env.valid_action_mask(), rng)
            encoded_reachable = encode_states(
                states, torch.device("cpu"), state_encoding,
                score_table=table_t, wide_norm=wide,
                include_current_line_scores=include_current_line_scores,
            )
            predicted_rewards = encoded_reachable[:, 50:75][
                torch.arange(env.n_envs), torch.from_numpy(actions)
            ] * ceiling
            if include_current_line_scores:
                # Score corrente (prima dell'azione) della riga di ciascuna
                # azione: deve coincidere con lo score ottenuto calcolando
                # il punteggio di quella riga direttamente con la stessa tabella.
                row_ids = torch.from_numpy(actions // GRID_SIZE)
                grid_3d = torch.from_numpy(states[:, :25].reshape(-1, GRID_SIZE, GRID_SIZE))
                current_row_all = score_lines_torch(grid_3d, table_t)
                expected_current_row = current_row_all[torch.arange(env.n_envs), row_ids] / line_ceiling
                got_current_row = encoded_reachable[:, current_block_offset : current_block_offset + 25][
                    torch.arange(env.n_envs), torch.from_numpy(actions)
                ]
                if not torch.allclose(got_current_row, expected_current_row, atol=1e-3):
                    max_error = float((got_current_row - expected_current_row).abs().max())
                    raise AssertionError(
                        f"Feature score-corrente-riga errata sotto tabella "
                        f"{table.tolist()}; errore massimo={max_error}"
                    )
            _, actual_rewards, _ = env.step(actions)
            expected = torch.from_numpy(actual_rewards.astype(np.float32, copy=False))
            if not torch.allclose(predicted_rewards, expected, atol=1e-3):
                max_error = float((predicted_rewards - expected).abs().max())
                raise AssertionError(
                    f"Feature reward immediato {state_encoding} errata sotto "
                    f"tabella {table.tolist()}; errore massimo={max_error}"
                )
        if not torch.allclose(encoded[2, 25:50], torch.zeros(N_CELLS)):
            raise AssertionError("Uno stato terminale non deve avere azioni valide")
        if not torch.isclose(encoded[2, 161], torch.tensor(1.0)):
            raise AssertionError(f"Turno terminale {state_encoding} errato")
        if wide:
            # 'encoded' in cima alla funzione è stato calcolato con wide_norm di
            # default (False): la tabella è quella di default ma il massimale è
            # TIGHT, non WIDE.
            score_offset = ENGINEERED184_STATE_SIZE + (
                CURRENT_LINE_SCORES_SIZE if include_current_line_scores else 0
            )
            score_block = encoded[:, score_offset : score_offset + SCORE_TABLE_SIZE]
            default_norm = (
                torch.from_numpy(DEFAULT_SCORE_TABLE.astype(np.float32)) / LINE_SCORE_CEILING_TIGHT
            )
            if not torch.allclose(score_block, default_norm.unsqueeze(0).expand(3, -1), atol=1e-5):
                raise AssertionError("Blocco di condizionamento sullo score errato in engineered192")


def validate_symmetry_augmentation(num_trials: int = 40, seed: int = 555) -> None:
    """Verifica che le 8 trasformazioni D4 lascino invariato lo score totale
    su griglie casuali (anche parzialmente riempite) e con tabelle di scoring
    sia di default sia randomizzate, e che apply_symmetry_to_actions tracci
    correttamente dove finisce la cella trasformata."""
    rng = np.random.default_rng(seed)
    possible_values = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.uint8)

    def total_score_of_grid(grid_5x5: np.ndarray, score_table: np.ndarray) -> int:
        env = FastVectorKnister(1, np.random.default_rng(0), score_table=score_table)
        total = 0
        for r in range(GRID_SIZE):
            total += int(env._score_lines(grid_5x5[r : r + 1, :])[0])
        for c in range(GRID_SIZE):
            total += int(env._score_lines(grid_5x5[:, c].reshape(1, -1))[0])
        main = np.array([grid_5x5[i, i] for i in range(GRID_SIZE)]).reshape(1, -1)
        anti = np.array([grid_5x5[i, GRID_SIZE - 1 - i] for i in range(GRID_SIZE)]).reshape(1, -1)
        total += DIAGONAL_MULTIPLIER * int(env._score_lines(main)[0])
        total += DIAGONAL_MULTIPLIER * int(env._score_lines(anti)[0])
        return total

    for trial in range(num_trials):
        grid_flat = rng.choice(possible_values, size=25)
        table = DEFAULT_SCORE_TABLE if trial % 2 == 0 else rng.integers(1, 20, size=8).astype(np.int32)
        base_score = total_score_of_grid(grid_flat.reshape(GRID_SIZE, GRID_SIZE), table)
        for g in range(N_D4):
            transformed = grid_flat[D4_FORWARD_PERM[g]].reshape(GRID_SIZE, GRID_SIZE)
            if total_score_of_grid(transformed, table) != base_score:
                raise AssertionError(
                    f"Simmetria D4 gruppo={g} non preserva lo score (trial={trial})"
                )
    for _ in range(20):
        a = int(rng.integers(0, 25))
        g = int(rng.integers(0, N_D4))
        ta = int(apply_symmetry_to_actions(np.array([a]), np.array([g]))[0])
        if D4_FORWARD_PERM[g][ta] != a:
            raise AssertionError(f"Trasformazione azione inconsistente: gruppo={g}, azione={a}")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesto ma non disponibile")
    return device


def epsilon_by_step(cfg: Config, env_steps: int) -> float:
    total_steps = cfg.episodes * 25
    decay_steps = max(1, int(total_steps * cfg.epsilon_decay_fraction))
    progress = min(1.0, env_steps / decay_steps)
    return cfg.epsilon_start + progress * (cfg.epsilon_end - cfg.epsilon_start)


def lr_by_updates(cfg: Config, gradient_updates: int) -> float:
    """Learning rate decrescente (opzionale). cfg.lr_end=None riproduce
    esattamente il comportamento originale (lr costante).

    Con lr_end impostato: lr resta fisso a cfg.lr fino a
    cfg.lr_decay_start_fraction degli aggiornamenti gradiente totali stimati
    (stessa stima di per_beta_by_updates), poi decresce linearmente fino a
    cfg.lr_end alla frazione cfg.lr_decay_fraction, poi resta fissa a lr_end.
    Con lr_decay_start_fraction=0.0 il decadimento parte da subito (comportamento
    della primissima versione di questa funzione — vedi changelog: su un run
    lungo può ridurre il lr proprio nella finestra di crescita più forte,
    prima che la rete abbia sfruttato l'esplorazione già fatta. Per un vero
    tapering finale, impostare lr_decay_start_fraction alto, es. 0.8).
    """
    if cfg.lr_end is None:
        return cfg.lr
    total_rounds = max(1, -(-cfg.episodes // cfg.n_envs))  # ceil division
    total_updates_estimate = max(1, 25 * total_rounds * cfg.updates_per_vector_step)
    start_updates = int(total_updates_estimate * cfg.lr_decay_start_fraction)
    end_updates = max(start_updates + 1, int(total_updates_estimate * cfg.lr_decay_fraction))
    if gradient_updates <= start_updates:
        return cfg.lr
    progress = min(1.0, (gradient_updates - start_updates) / (end_updates - start_updates))
    return cfg.lr + progress * (cfg.lr_end - cfg.lr)


def per_beta_by_updates(cfg: Config, gradient_updates: int) -> float:
    """Incrementa gradualmente il parametro beta (esponente di importance sampling
    per PER) da per_beta_start a per_beta_end, linearmente sulla frazione di update
    gradiente completati (stima: 25 * (episodes/n_envs) * updates_per_vector_step,
    assumendo che l'apprendimento sia attivo per la quasi totalità del run —
    approssimazione ragionevole dato che learning_starts è tipicamente una
    piccola frazione di episodes). beta=1 a fine training annulla il bias
    introdotto dal campionamento non uniforme.
    """
    total_rounds = max(1, -(-cfg.episodes // cfg.n_envs))  # divisione con arrotondamento per eccesso
    total_updates_estimate = max(1, 25 * total_rounds * cfg.updates_per_vector_step)
    progress = min(1.0, gradient_updates / total_updates_estimate)
    return cfg.per_beta_start + progress * (cfg.per_beta_end - cfg.per_beta_start)


def random_valid_actions(valid_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Punteggi casuali + argmin fornisce una cella libera casuale uniforme per ogni riga.
    random_scores = rng.random(valid_mask.shape)
    random_scores[~valid_mask] = 2.0
    return random_scores.argmin(axis=1).astype(np.int64)


def collect_n_step_transitions(
    traj_states: np.ndarray,
    traj_actions: np.ndarray,
    traj_rewards: np.ndarray,
    n_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Costruisce transizioni (1-step o n-step) da una traiettoria completa.

    Args:
      traj_states:  [26, n_envs, RAW_STATE_SIZE] uint8. traj_states[i] è lo
                    stato dopo i azioni (traj_states[0] = stato iniziale).
      traj_actions: [25, n_envs] azioni scelte a ciascun passo.
      traj_rewards: [25, n_envs] reward reali ricevuti a ciascun passo.
      n_step: ampiezza della finestra di ritorno. n_step=1 riproduce
              esattamente le transizioni a singolo step della versione
              precedente.

    Con gamma=1 (l'unico usato in questo progetto: verifica
    formale) il ritorno a n-step è la somma semplice dei reward reali nella
    finestra, senza pesi esponenziali. Poiché l'orizzonte è sempre esattamente
    25 e fisso (nessuna terminazione anticipata), la finestra si accorcia in
    modo deterministico vicino alla fine dell'episodio invece di dover gestire
    episodi di lunghezza variabile.

    Returns: (states, actions, returns, next_states, dones), ciascuno con la
    prima dimensione 25 (poi va appiattita insieme a n_envs prima di
    ReplayBuffer.add_batch).
    """
    if n_step < 1:
        raise ValueError("n_step deve essere >= 1")
    steps, n_envs = traj_actions.shape  # 25, n_envs
    cumsum = np.concatenate(
        [
            np.zeros((1, n_envs), dtype=np.float64),
            np.cumsum(traj_rewards, axis=0, dtype=np.float64),
        ],
        axis=0,
    )  # [26, n_envs]
    t_idx = np.arange(steps)
    target_idx = np.minimum(t_idx + n_step, steps)  # clip a 25 (fine episodio)
    returns = (cumsum[target_idx] - cumsum[t_idx]).astype(np.int32)  # [25, n_envs]
    dones = target_idx == steps  # finestra che tocca/supera il termine

    out_states = traj_states[t_idx]  # [25, n_envs, RAW_STATE_SIZE]
    out_next_states = traj_states[target_idx]  # [25, n_envs, RAW_STATE_SIZE]
    dones_full = np.broadcast_to(dones[:, None], (steps, n_envs)).copy()
    return out_states, traj_actions, returns, out_next_states, dones_full


def select_actions(
    states_u8: np.ndarray,
    epsilon: float,
    policy: nn.Module,
    device: torch.device,
    rng: np.random.Generator,
    state_encoding: str,
    score_table: Optional[torch.Tensor] = None,
    wide_norm: bool = False,
    dueling_mask_aware: bool = True,
    include_current_line_scores: bool = False,
) -> np.ndarray:
    valid = states_u8[:, :25] == 0
    n = states_u8.shape[0]
    explore = rng.random(n) < epsilon
    actions = np.empty(n, dtype=np.int64)

    if np.any(explore):
        actions[explore] = random_valid_actions(valid[explore], rng)

    exploit = ~explore
    if np.any(exploit):
        # Un solo trasferimento compatto e un solo forward pass in batch.
        x = encode_states(
            states_u8[exploit], device, state_encoding,
            score_table=score_table, wide_norm=wide_norm,
            include_current_line_scores=include_current_line_scores,
        )
        valid_t = torch.from_numpy(valid[exploit]).to(device=device)
        with torch.inference_mode():
            q = policy(x, valid_t if dueling_mask_aware else None)
            q.masked_fill_(~valid_t, torch.finfo(q.dtype).min)
            greedy = q.argmax(dim=1).cpu().numpy()
        actions[exploit] = greedy

    return actions


def train_one_batch(
    policy: nn.Module,
    target: nn.Module,
    policy_parameters,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    grad_clip: float,
    device: torch.device,
    state_encoding: str,
    wide_norm: bool = False,
    dueling_mask_aware: bool = True,
    n_step: int = 1,
    include_current_line_scores: bool = False,
    beta: float = 1.0,
    symmetry_augmentation: bool = False,
    symmetry_rng: Optional[np.random.Generator] = None,
) -> float:
    (
        states_np, actions_np, rewards_np, next_states_np, dones_np, score_tables_np,
        replay_indices, is_weights_np,
    ) = replay.sample(batch_size, beta=beta)

    if symmetry_augmentation:
        # Una trasformazione D4
        # casuale e indipendente per ciascun campione del batch (compresa
        # l'identità, 1/8 delle volte): stesso reward per costruzione (score
        # Knister invariante per simmetria della griglia, verificato in
        # validate_symmetry_augmentation), quindi non serve toccare rewards.
        group_idx = symmetry_rng.integers(0, N_D4, size=batch_size)
        states_np = apply_symmetry_to_states(states_np, group_idx)
        next_states_np = apply_symmetry_to_states(next_states_np, group_idx)
        actions_np = apply_symmetry_to_actions(actions_np, group_idx)

    # Tabella per-transizione (una riga per campione, vedi ReplayBuffer): un
    # minibatch può mescolare round diversi con tabelle diverse quando
    # randomize_scores=True. Con randomize_scores=False è comunque sempre
    # DEFAULT_SCORE_TABLE, quindi score_lines_torch la tratta correttamente
    # come caso degenere di [batch,8] tutte uguali.
    score_table = torch.from_numpy(score_tables_np.astype(np.float32)).to(device=device)

    states = encode_states(
        states_np, device, state_encoding, score_table=score_table, wide_norm=wide_norm,
        include_current_line_scores=include_current_line_scores,
    )
    next_states = encode_states(
        next_states_np, device, state_encoding, score_table=score_table, wide_norm=wide_norm,
        include_current_line_scores=include_current_line_scores,
    )
    actions = torch.from_numpy(actions_np.astype(np.int64, copy=False)).to(device=device)
    rewards = torch.from_numpy(rewards_np.astype(np.float32, copy=False)).to(device=device)
    dones = torch.from_numpy(dones_np).to(device=device)
    is_weights = torch.from_numpy(is_weights_np).to(device=device)

    valid = next_states_np[:, :25] == 0
    valid_t = torch.from_numpy(states_np[:, :25] == 0).to(device=device)
    q_all = policy(states, valid_t if dueling_mask_aware else None)
    q_selected = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Double DQN: la policy network seleziona l'azione successiva, la target invece la valuta
    with torch.no_grad():
        next_valid_t = torch.from_numpy(valid).to(device=device)
        next_policy_q = policy(next_states, next_valid_t if dueling_mask_aware else None)
        next_policy_q.masked_fill_(~next_valid_t, torch.finfo(next_policy_q.dtype).min)
        next_actions = next_policy_q.argmax(dim=1, keepdim=True)
        next_target_q = target(
            next_states, next_valid_t if dueling_mask_aware else None
        ).gather(1, next_actions).squeeze(1)
        next_target_q.masked_fill_(dones, 0.0)
        # rewards è già la somma (non scontata) su n_step reward reali quando
        # n_step>1 (vedi collect_n_step_transitions); qui serve solo scontare
        # il bootstrap di n_step passi. gamma**n_step == gamma per n_step=1,
        # quindi il caso standard è invariato.
        targets = rewards + (gamma ** n_step) * next_target_q

    # Loss non ridotta: serve sia per il peso di importance sampling (is_weights
    # è un vettore di soli 1 quando non prioritizzato, quindi
    # equivale alla media semplice di prima) sia per il TD-error da usare
    # nell'aggiornamento delle priorità.
    per_sample_loss = F.smooth_l1_loss(q_selected, targets, reduction="none")
    loss = (per_sample_loss * is_weights).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy_parameters, grad_clip)
    optimizer.step()

    with torch.no_grad():
        td_errors = (targets - q_selected).detach().cpu().numpy()
    replay.update_priorities(replay_indices, td_errors)

    return float(loss.detach().cpu())


# -----------------------------------------------------------------------------
# Verifica di correttezza rispetto all'API ufficiale
# -----------------------------------------------------------------------------


def validate_fast_environment(num_games: int = 64, seed: int = 1234) -> None:
    rng = np.random.default_rng(seed)
    rolls = (
        rng.integers(1, 7, size=(num_games, 25), dtype=np.uint8)
        + rng.integers(1, 7, size=(num_games, 25), dtype=np.uint8)
    )

    fast = FastVectorKnister(num_games, np.random.default_rng(seed + 1))
    fast.reset(first_rolls=rolls[:, 0])

    official = [KnisterGame() for _ in range(num_games)]
    for i, game in enumerate(official):
        game.new_game()
        game.set_current_roll(int(rolls[i, 0]))

    action_rng = np.random.default_rng(seed + 2)
    for step in range(25):
        valid = fast.valid_action_mask()
        actions = random_valid_actions(valid, action_rng)
        next_rolls = rolls[:, step + 1] if step < 24 else None
        _, fast_rewards, done = fast.step(actions, forced_next_rolls=next_rolls)

        for i, game in enumerate(official):
            game.choose_action(int(actions[i]))
            if game.get_last_reward() != int(fast_rewards[i]):
                raise AssertionError(
                    f"Discrepanza della ricompensa nella partita={i}, step={step}: "
                    f"official={game.get_last_reward()}, fast={fast_rewards[i]}"
                )
            if step < 24:
                game.set_current_roll(int(rolls[i, step + 1]))
            if not np.array_equal(game.get_grid().reshape(-1), fast.grids[i]):
                raise AssertionError(f"Grid mismatch game={i}, step={step}")

        if bool(done[0]) != (step == 24):
            raise AssertionError("Discrepanza del flag 'done'")

    for i, game in enumerate(official):
        if game.get_total_reward() != int(fast.total_scores[i]):
            raise AssertionError(
                f"Discrepanza del punteggio finale nella partita={i}: "
                f"official={game.get_total_reward()}, fast={fast.total_scores[i]}"
            )


# -----------------------------------------------------------------------------
# Valutazione
# -----------------------------------------------------------------------------


def greedy_actions(
    states_u8: np.ndarray,
    model: nn.Module,
    device: torch.device,
    state_encoding: str,
    wide_norm: bool = False,
    dueling_mask_aware: bool = True,
    include_current_line_scores: bool = False,
) -> np.ndarray:
    """Seleziona le azioni valide greedy tramite un singolo forward pass della rete in batch.

    `score_table` non è un parametro qui: la valutazione misura sempre le
    prestazioni sotto la tabella VERA (DEFAULT_SCORE_TABLE, usata di default
    da encode_states quando score_table=None), indipendentemente da
    randomize_scores in training — altrimenti il punteggio riportato non
    sarebbe più confrontabile con le run precedenti.
    `wide_norm` invece deve riflettere la scala su cui il modello è
    stato allenato (cfg.randomize_scores), non la tabella in uso. Nessuna
    augmentation da simmetria qui: la valutazione è greedy e deterministica
    sullo stato vero, non uno scopo di training.
    """
    valid = states_u8[:, :25] == 0
    x = encode_states(
        states_u8, device, state_encoding, wide_norm=wide_norm,
        include_current_line_scores=include_current_line_scores,
    )
    valid_t = torch.from_numpy(valid).to(device=device)
    with torch.inference_mode():
        q = model(x, valid_t if dueling_mask_aware else None)
        q.masked_fill_(~valid_t, torch.finfo(q.dtype).min)
        return q.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)


def evaluate_vectorized(
    model: nn.Module,
    device: torch.device,
    games_count: int = 500,
    seed: int = 9876,
    batch_size: int = 1024,
    state_encoding: str = "onehot",
    wide_norm: bool = False,
    dueling_mask_aware: bool = True,
    include_current_line_scores: bool = False,
) -> tuple[float, float]:
    """Valuta in modo greedy utilizzando lo stesso ambiente vettorializzato e verificato dell'addestramento.

    L'ambiente usa sempre DEFAULT_SCORE_TABLE (FastVectorKnister di default):
    il punteggio riportato è sempre sotto la tabella vera, così resta
    confrontabile con le run precedenti indipendentemente da randomize_scores.
    """
    if games_count <= 0:
        raise ValueError("games_count deve essere positivo")
    if batch_size <= 0:
        raise ValueError("batch_size deve essere positivo")

    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    scores_parts: list[np.ndarray] = []
    completed = 0

    try:
        while completed < games_count:
            current_n = min(batch_size, games_count - completed)
            print(
                f"Valutazione vettorizzata: avvio batch da {current_n} match "
                f"({completed}/{games_count} già completati)",
                flush=True,
            )
            env = FastVectorKnister(current_n, rng)
            states = env.observe()

            for _ in range(FastVectorKnister.N_CELLS):
                actions = greedy_actions(
                    states, model, device, state_encoding,
                    wide_norm=wide_norm, dueling_mask_aware=dueling_mask_aware,
                    include_current_line_scores=include_current_line_scores,
                )
                states, _, _ = env.step(actions)

            scores_parts.append(env.total_scores.astype(np.float32, copy=True))
            completed += current_n
            print(
                f"Valutazione vettorizzata: {completed}/{games_count} match completati",
                flush=True,
            )
    finally:
        if was_training:
            model.train()

    scores = np.concatenate(scores_parts)
    return float(scores.mean()), float(scores.std())


def evaluate_official(
    model: nn.Module,
    device: torch.device,
    games_count: int = 500,
    seed: int = 9876,
    progress_every: int = 50,
    state_encoding: str = "onehot",
    wide_norm: bool = False,
    dueling_mask_aware: bool = True,
    include_current_line_scores: bool = False,
) -> tuple[float, float]:
    """Valuta con api.KnisterGame e stampa l'avanzamento dopo ogni blocco di partite.

    L'inferenza della rete è comunque eseguita in batch all'interno di ciascun blocco,
    mentre le chiamate all'ambiente Python ufficiale rimangono sequenziali. 
    Questa modalità è più lenta ma utile come controllo di compatibilità finale.
    """
    if games_count <= 0:
        raise ValueError("games_count deve essere positivo")
    if progress_every <= 0:
        progress_every = games_count

    py_state = random.getstate()
    was_training = model.training
    random.seed(seed)
    model.eval()
    scores_parts: list[np.ndarray] = []
    completed = 0

    try:
        while completed < games_count:
            current_n = min(progress_every, games_count - completed)
            games = [KnisterGame() for _ in range(current_n)]
            for game in games:
                game.new_game()

            with torch.inference_mode():
                for _ in range(25):
                    states = np.empty((current_n, 26), dtype=np.uint8)
                    valid = np.empty((current_n, 25), dtype=np.bool_)
                    for i, game in enumerate(games):
                        grid = game.get_grid().reshape(-1)
                        states[i, :25] = grid
                        states[i, 25] = game.get_current_roll()
                        valid[i] = grid == 0

                    x = encode_states(
                        states, device, state_encoding, wide_norm=wide_norm,
                        include_current_line_scores=include_current_line_scores,
                    )
                    valid_t = torch.from_numpy(valid).to(device=device)
                    q = model(x, valid_t if dueling_mask_aware else None)
                    q.masked_fill_(~valid_t, torch.finfo(q.dtype).min)
                    actions = q.argmax(dim=1).cpu().numpy()

                    for game, action in zip(games, actions):
                        game.choose_action(int(action))

            scores_parts.append(
                np.fromiter(
                    (game.get_total_reward() for game in games),
                    dtype=np.float32,
                    count=current_n,
                )
            )
            completed += current_n
            print(
                f"Valutazione: {completed}/{games_count} match completati",
                flush=True,
            )
    finally:
        random.setstate(py_state)
        if was_training:
            model.train()

    scores = np.concatenate(scores_parts)
    return float(scores.mean()), float(scores.std())


def run_evaluation(
    model: nn.Module,
    device: torch.device,
    cfg: Config,
    *,
    seed: Optional[int] = None,
    games_count: Optional[int] = None,
    label: str = "validazione",
) -> tuple[float, float]:
    """Esegue la valutazione e mostra anche l'incertezza sulla media."""
    actual_seed = cfg.eval_seed if seed is None else int(seed)
    actual_games = cfg.eval_games if games_count is None else int(games_count)
    if actual_games <= 0:
        raise ValueError("games_count deve essere positivo")

    mode_label = (
        "vettorizzata"
        if cfg.eval_mode == "vectorized"
        else "API ufficiale sequenziale a blocchi"
    )
    print(
        f"\nInizio {label}... modalità={mode_label}, match={actual_games}, seed={actual_seed}",
        flush=True,
    )
    eval_start = time.perf_counter()

    # wide_norm segue randomize_scores: è la scala su cui il modello è stato
    # allenato, non la tabella (sempre quella vera in valutazione, vedi
    # greedy_actions/evaluate_vectorized/evaluate_official).
    if cfg.eval_mode == "vectorized":
        mean, std = evaluate_vectorized(
            model=model,
            device=device,
            games_count=actual_games,
            seed=actual_seed,
            batch_size=cfg.eval_batch_size,
            state_encoding=cfg.state_encoding,
            wide_norm=cfg.randomize_scores,
            dueling_mask_aware=cfg.dueling_mask_aware,
            include_current_line_scores=cfg.include_current_line_scores,
        )
    elif cfg.eval_mode == "official":
        mean, std = evaluate_official(
            model=model,
            device=device,
            games_count=actual_games,
            seed=actual_seed,
            progress_every=cfg.eval_progress_every,
            state_encoding=cfg.state_encoding,
            wide_norm=cfg.randomize_scores,
            dueling_mask_aware=cfg.dueling_mask_aware,
            include_current_line_scores=cfg.include_current_line_scores,
        )
    else:
        raise ValueError(
            f"eval_mode non valido: {cfg.eval_mode!r}. "
            "Usare 'vectorized' oppure 'official'."
        )

    elapsed = time.perf_counter() - eval_start
    sem = std / math.sqrt(actual_games)
    ci95 = 1.96 * sem
    print(
        f"Fine {label} | tempo: {format_duration(elapsed)} | "
        f"media={mean:.2f}, deviazione standard={std:.2f}, "
        f"IC95 media≈[{mean - ci95:.2f}, {mean + ci95:.2f}]\n",
        flush=True,
    )
    return mean, std


# -----------------------------------------------------------------------------
# Monitoraggio dell'avanzamento
# -----------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Formatta una durata come DDg HH:MM:SS o HH:MM:SS."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"

    total_seconds = int(round(seconds))
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)

    if days > 0:
        return f"{days}g {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(
    *,
    episodes_done: int,
    total_episodes: int,
    interval_episodes: int,
    interval_seconds: float,
    elapsed_seconds: float,
    current_speed: float,
    epsilon: float,
    mean_score: float,
    mean_loss: float,
    gradient_updates: int,
    current_lr: Optional[float] = None,
) -> None:
    """Stampa un report di avanzamento completo, compatibile con Colab e con ETA."""
    percentage = 100.0 * episodes_done / max(total_episodes, 1)
    interval_speed = interval_episodes / max(interval_seconds, 1e-9)
    overall_speed = episodes_done / max(elapsed_seconds, 1e-9)
    remaining_episodes = max(0, total_episodes - episodes_done)
    eta_seconds = remaining_episodes / current_speed if current_speed > 0 else math.inf

    if math.isfinite(eta_seconds):
        estimated_finish = datetime.now() + timedelta(seconds=eta_seconds)
        finish_text = estimated_finish.strftime("%d/%m/%Y %H:%M:%S")
    else:
        finish_text = "non disponibile"

    lr_part = f" | lr={current_lr:.2e}" if current_lr is not None else ""
    print(
        f"[{percentage:6.2f}%] "
        f"Episodi {episodes_done:,}/{total_episodes:,} | "
        f"ultimi {interval_episodes:,}: {format_duration(interval_seconds)} "
        f"({interval_speed:,.0f} ep/s) | "
        f"totale: {format_duration(elapsed_seconds)} | "
        f"media: {overall_speed:,.0f} ep/s | "
        f"ETA: {format_duration(eta_seconds)} | "
        f"fine stimata: {finish_text} | "
        f"eps={epsilon:.3f} | score10k={mean_score:.2f} | "
        f"loss={mean_loss:.5f} | update={gradient_updates:,}{lr_part}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def save_model(
    model: DQN,
    cfg: Config,
    episodes_done: int,
    path_value: str,
    *,
    checkpoint_type: str,
    eval_mean: Optional[float] = None,
    eval_std: Optional[float] = None,
) -> None:
    """Salva pesi e metadati senza confondere ultimo e miglior checkpoint."""
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    metadata = {
        "checkpoint_type": checkpoint_type,
        "episodes_done": episodes_done,
        "eval_mean": eval_mean,
        "eval_std": eval_std,
        "raw_state_size": RAW_STATE_SIZE,
        "input_size": state_input_size(cfg.state_encoding, cfg.include_current_line_scores),
        "state_encoding": cfg.state_encoding,
        "output_size": 25,
        "hidden_size": cfg.hidden_size,
        "network_type": cfg.network_type,
        "config": asdict(cfg),
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_model_weights(path_value: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Carica in modo compatibile con versioni PyTorch vecchie e nuove."""
    try:
        return torch.load(path_value, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path_value, map_location=device)


def train(cfg: Config) -> None:
    if cfg.batch_size > cfg.replay_capacity:
        raise ValueError("batch_size non può eccedere replay_capacity")
    if cfg.eval_games <= 0:
        raise ValueError("eval_games deve essere positivo")
    if cfg.final_eval_games < 0:
        raise ValueError("final_eval_games non può essere negativo")
    if cfg.network_type not in {"mlp", "dueling"}:
        raise ValueError("network_type deve essere 'mlp' oppure 'dueling'")
    if cfg.eval_batch_size <= 0:
        raise ValueError("eval_batch_size deve essere positivo")
    if cfg.eval_progress_every <= 0:
        raise ValueError("eval_progress_every deve essere positivo")
    if Path(cfg.save_path).resolve() == Path(cfg.best_save_path).resolve():
        raise ValueError("save_path e best_save_path devono essere diversi")
    if cfg.n_step < 1 or cfg.n_step > 25:
        raise ValueError("n_step deve essere tra 1 e 25 (l'orizzonte è sempre 25 passi)")
    if cfg.randomize_scores and not (0 < cfg.score_jitter_low <= cfg.score_jitter_high):
        raise ValueError(
            "score_jitter_low/high non validi: serve 0 < jitter_low <= jitter_high"
        )
    if cfg.prioritized_replay:
        if not (0.0 <= cfg.per_alpha <= 1.0):
            raise ValueError("per_alpha deve essere in [0, 1]")
        if not (0.0 < cfg.per_beta_start <= cfg.per_beta_end <= 1.0):
            raise ValueError("serve 0 < per_beta_start <= per_beta_end <= 1")
        if cfg.per_epsilon <= 0:
            raise ValueError("per_epsilon deve essere positivo")
    if cfg.lr_end is not None:
        if cfg.lr_end <= 0 or cfg.lr <= 0:
            raise ValueError("lr e lr_end devono essere positivi")
        if not (0.0 <= cfg.lr_decay_start_fraction < cfg.lr_decay_fraction <= 1.0):
            raise ValueError(
                "serve 0 <= lr_decay_start_fraction < lr_decay_fraction <= 1"
            )
    input_size = state_input_size(cfg.state_encoding, cfg.include_current_line_scores)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = choose_device(cfg.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, cfg.cpu_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    rng = np.random.default_rng(cfg.seed)
    replay_rng = np.random.default_rng(cfg.seed + 1)
    symmetry_rng = np.random.default_rng(cfg.seed + 3)

    print("Verifica equivalenza ambiente vettorizzato/API ufficiale...", flush=True)
    validate_fast_environment()
    print("OK: reward, griglie e punteggi finali coincidono.", flush=True)
    print(
        f"Verifica encoder stato: modalità={cfg.state_encoding}, input_size={input_size}"
        f"{' (+75 score correnti riga/colonna/diagonale)' if cfg.include_current_line_scores else ''}...",
        flush=True,
    )
    validate_state_encoder(cfg.state_encoding, cfg.include_current_line_scores)
    print("OK: encoder dello stato valido.", flush=True)
    if cfg.symmetry_augmentation:
        validate_symmetry_augmentation()
        print("OK: simmetrie D4 verificate (score invariante).", flush=True)

    policy_raw = DQN(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        network_type=cfg.network_type,
    ).to(device)
    target_raw = DQN(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        network_type=cfg.network_type,
    ).to(device)
    target_raw.load_state_dict(policy_raw.state_dict())
    target_raw.eval()

    policy: nn.Module = policy_raw
    target: nn.Module = target_raw
    if cfg.compile_model:
        if not hasattr(torch, "compile"):
            print("torch.compile non disponibile: continuo in eager mode.")
        else:
            print("Compilazione dei forward con torch.compile...")
            policy = torch.compile(policy_raw, mode="reduce-overhead", dynamic=True)
            target = torch.compile(target_raw, mode="reduce-overhead", dynamic=True)

    optimizer = torch.optim.Adam(policy_raw.parameters(), lr=cfg.lr)
    if cfg.prioritized_replay:
        replay: ReplayBuffer = PrioritizedReplayBuffer(
            cfg.replay_capacity, replay_rng, alpha=cfg.per_alpha, epsilon=cfg.per_epsilon
        )
    else:
        replay = ReplayBuffer(cfg.replay_capacity, replay_rng)

    env_steps = 0
    gradient_updates = 0
    current_lr = cfg.lr
    episodes_done = 0
    batch_index = 0
    last_eval_episodes = -1
    last_eval_result: Optional[tuple[float, float]] = None
    best_eval_mean = -math.inf
    best_eval_std = math.nan
    best_eval_episodes = 0
    # score10k (nel log) è la media recente dei punteggi raccolti con policy
    # epsilon-greedy: già di per sé non va confrontata con la valutazione
    # greedy (vedi run_evaluation). Con randomize_scores=True è ANCHE una
    # media su round con tabelle di scoring diverse tra loro: utile solo come
    # segnale approssimativo "il training sta procedendo", non come stima di
    # punteggio. Il numero comparabile alle run precedenti resta sempre
    # quello di run_evaluation, che usa la tabella vera indipendentemente da
    # randomize_scores.
    recent_scores = deque(maxlen=10_000)
    recent_losses = deque(maxlen=1_000)
    start_time = time.perf_counter()
    last_log = start_time
    last_log_episodes = 0
    # Campioni recenti (episodi, tempo) per un'ETA basata sulla velocità attuale,
    # più stabile del solo ultimo intervallo.
    progress_samples = deque(maxlen=max(2, cfg.progress_window + 1))
    progress_samples.append((0, start_time))
    next_log_episode = (
        min(cfg.log_every_episodes, cfg.episodes)
        if cfg.log_every_episodes > 0
        else None
    )

    print(
        f"Training: device={device}, episodes={cfg.episodes:,}, n_envs={cfg.n_envs}, "
        f"batch={cfg.batch_size}, replay={cfg.replay_capacity:,}, "
        f"stato={cfg.state_encoding}({input_size}), rete={cfg.network_type}, hidden={cfg.hidden_size}, "
        f"updates/step={cfg.updates_per_vector_step}, lr={cfg.lr:g}, "
        f"eps={cfg.epsilon_start:.2f}->{cfg.epsilon_end:.2f} in {cfg.epsilon_decay_fraction:.2f}, "
        f"report ogni ~{cfg.log_every_episodes:,} episodi"
        if cfg.log_every_episodes > 0
        else
        f"Training: device={device}, episodes={cfg.episodes:,}, n_envs={cfg.n_envs}, "
        f"batch={cfg.batch_size}, replay={cfg.replay_capacity:,}, "
        f"stato={cfg.state_encoding}({input_size}), rete={cfg.network_type}, hidden={cfg.hidden_size}, "
        f"updates/step={cfg.updates_per_vector_step}, lr={cfg.lr:g}, "
        f"eps={cfg.epsilon_start:.2f}->{cfg.epsilon_end:.2f} in {cfg.epsilon_decay_fraction:.2f}, "
        f"report ogni {cfg.log_every_batches} batch vettoriali",
        flush=True,
    )
    print(
        f"Conformità/ablation: randomize_scores={cfg.randomize_scores}"
        + (
            f" (jitter [{cfg.score_jitter_low:g}, {cfg.score_jitter_high:g}])"
            if cfg.randomize_scores else ""
        )
        + f", dueling_mask_aware={cfg.dueling_mask_aware}, n_step={cfg.n_step}, "
        + f"current_line_scores={cfg.include_current_line_scores}, "
        + f"prioritized_replay={cfg.prioritized_replay}"
        + (
            f" (alpha={cfg.per_alpha:g}, beta {cfg.per_beta_start:g}->{cfg.per_beta_end:g})"
            if cfg.prioritized_replay else ""
        )
        + f", symmetry_augmentation={cfg.symmetry_augmentation}"
        + (
            f", lr_decay {cfg.lr:g}->{cfg.lr_end:g} in [{cfg.lr_decay_start_fraction:.2f}, {cfg.lr_decay_fraction:.2f}]"
            if cfg.lr_end is not None else ""
        ),
        flush=True,
    )

    while episodes_done < cfg.episodes:
        current_n = min(cfg.n_envs, cfg.episodes - episodes_done)

        if cfg.randomize_scores:
            score_table_np = sample_score_table(
                rng, jitter_low=cfg.score_jitter_low, jitter_high=cfg.score_jitter_high
            )
        else:
            score_table_np = DEFAULT_SCORE_TABLE
        score_table_shared = torch.from_numpy(score_table_np.astype(np.float32)).to(device=device)

        env = FastVectorKnister(current_n, rng, score_table=score_table_np)
        states = env.observe()

        # Traiettoria completa del round, necessaria per costruire ritorni a
        # n-step (n_step=1 la usa comunque, per un unico percorso di codice).
        traj_states = np.empty((26, current_n, RAW_STATE_SIZE), dtype=np.uint8)
        traj_actions = np.empty((25, current_n), dtype=np.int64)
        traj_rewards = np.empty((25, current_n), dtype=np.int32)
        traj_states[0] = states

        for t in range(25):
            epsilon = epsilon_by_step(cfg, env_steps)
            actions = select_actions(
                states, epsilon, policy, device, rng, cfg.state_encoding,
                score_table=score_table_shared, wide_norm=cfg.randomize_scores,
                dueling_mask_aware=cfg.dueling_mask_aware,
                include_current_line_scores=cfg.include_current_line_scores,
            )
            next_states, rewards, dones = env.step(actions)

            traj_actions[t] = actions
            traj_rewards[t] = rewards
            traj_states[t + 1] = next_states

            states = next_states
            env_steps += current_n

        t_states, t_actions, t_returns, t_next_states, t_dones = collect_n_step_transitions(
            traj_states, traj_actions, traj_rewards, cfg.n_step
        )
        n_transitions = 25 * current_n
        score_tables_batch = np.broadcast_to(
            score_table_np, (n_transitions, SCORE_TABLE_SIZE)
        )
        replay.add_batch(
            t_states.reshape(n_transitions, RAW_STATE_SIZE),
            t_actions.reshape(n_transitions),
            t_returns.reshape(n_transitions),
            t_next_states.reshape(n_transitions, RAW_STATE_SIZE),
            t_dones.reshape(n_transitions),
            score_tables_batch,
        )

        if len(replay) >= max(cfg.learning_starts, cfg.batch_size):
            # 25 * updates_per_vector_step per round: stesso numero totale di
            # aggiornamenti gradiente della versione precedente (che ne faceva
            # updates_per_vector_step ad ognuno dei 25 passi), semplicemente
            # eseguiti dopo che l'intero round è stato spinto nel replay
            # invece che intrecciati passo-per-passo — necessario perché una
            # transizione a n-step non è disponibile finché la sua finestra
            # non si è chiusa.
            for _ in range(25 * cfg.updates_per_vector_step):
                beta = per_beta_by_updates(cfg, gradient_updates)
                current_lr = lr_by_updates(cfg, gradient_updates)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                loss = train_one_batch(
                    policy=policy,
                    target=target,
                    policy_parameters=policy_raw.parameters(),
                    optimizer=optimizer,
                    replay=replay,
                    batch_size=cfg.batch_size,
                    gamma=cfg.gamma,
                    grad_clip=cfg.grad_clip,
                    device=device,
                    state_encoding=cfg.state_encoding,
                    wide_norm=cfg.randomize_scores,
                    dueling_mask_aware=cfg.dueling_mask_aware,
                    n_step=cfg.n_step,
                    include_current_line_scores=cfg.include_current_line_scores,
                    beta=beta,
                    symmetry_augmentation=cfg.symmetry_augmentation,
                    symmetry_rng=symmetry_rng,
                )
                recent_losses.append(loss)
                gradient_updates += 1

                if gradient_updates % cfg.target_update_every == 0:
                    target_raw.load_state_dict(policy_raw.state_dict())

        episodes_done += current_n
        batch_index += 1
        recent_scores.extend(env.total_scores.astype(np.float32).tolist())

        if cfg.log_every_episodes > 0:
            should_log = (
                next_log_episode is not None and episodes_done >= next_log_episode
            ) or episodes_done == cfg.episodes
        else:
            should_log = (
                batch_index % max(1, cfg.log_every_batches) == 0
                or episodes_done == cfg.episodes
            )

        if should_log:
            now = time.perf_counter()
            interval = now - last_log
            eps_interval = episodes_done - last_log_episodes
            elapsed = now - start_time
            mean_score = float(np.mean(recent_scores)) if recent_scores else float("nan")
            mean_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")

            progress_samples.append((episodes_done, now))
            oldest_episodes, oldest_time = progress_samples[0]
            current_speed = (episodes_done - oldest_episodes) / max(
                now - oldest_time, 1e-9
            )

            print_progress(
                episodes_done=episodes_done,
                total_episodes=cfg.episodes,
                interval_episodes=eps_interval,
                interval_seconds=interval,
                elapsed_seconds=elapsed,
                current_speed=current_speed,
                epsilon=epsilon,
                mean_score=mean_score,
                mean_loss=mean_loss,
                gradient_updates=gradient_updates,
                current_lr=current_lr if cfg.lr_end is not None else None,
            )
            last_log = now
            last_log_episodes = episodes_done

            if cfg.log_every_episodes > 0 and next_log_episode is not None:
                while next_log_episode <= episodes_done:
                    next_log_episode += cfg.log_every_episodes

        if (
            cfg.eval_every_episodes > 0
            and episodes_done % cfg.eval_every_episodes < current_n
        ):
            mean, std = run_evaluation(policy, device, cfg, seed=cfg.eval_seed)
            last_eval_episodes = episodes_done
            last_eval_result = (mean, std)

            save_model(
                policy_raw,
                cfg,
                episodes_done,
                cfg.save_path,
                checkpoint_type="last",
                eval_mean=mean,
                eval_std=std,
            )
            print(f"Checkpoint LAST aggiornato: {cfg.save_path}", flush=True)

            if mean > best_eval_mean:
                previous_best = best_eval_mean
                best_eval_mean = mean
                best_eval_std = std
                best_eval_episodes = episodes_done
                save_model(
                    policy_raw,
                    cfg,
                    episodes_done,
                    cfg.best_save_path,
                    checkpoint_type="best",
                    eval_mean=mean,
                    eval_std=std,
                )
                old_text = "nessuno" if not math.isfinite(previous_best) else f"{previous_best:.2f}"
                print(
                    f"NUOVO MIGLIOR MODELLO: {mean:.2f} ± {std:.2f} "
                    f"(precedente: {old_text}) a {episodes_done:,} episodi",
                    flush=True,
                )
                print(f"Checkpoint BEST salvato: {cfg.best_save_path}", flush=True)

    if last_eval_episodes == episodes_done and last_eval_result is not None:
        mean, std = last_eval_result
        print(
            "Valutazione finale già eseguita all'ultimo checkpoint: "
            "riutilizzo il risultato senza ripetere i match.",
            flush=True,
        )
    else:
        mean, std = run_evaluation(policy, device, cfg, seed=cfg.eval_seed)

    # Garantisce che LAST rappresenti sempre esattamente la fine del training.
    save_model(
        policy_raw,
        cfg,
        episodes_done,
        cfg.save_path,
        checkpoint_type="last",
        eval_mean=mean,
        eval_std=std,
    )

    # Se la valutazione finale non era passata dal blocco periodico, può essere il best.
    if mean > best_eval_mean:
        best_eval_mean = mean
        best_eval_std = std
        best_eval_episodes = episodes_done
        save_model(
            policy_raw,
            cfg,
            episodes_done,
            cfg.best_save_path,
            checkpoint_type="best",
            eval_mean=mean,
            eval_std=std,
        )

    holdout_result: Optional[tuple[float, float]] = None
    if cfg.final_eval_games > 0:
        best_model = DQN(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            network_type=cfg.network_type,
        ).to(device)
        best_model.load_state_dict(load_model_weights(cfg.best_save_path, device))
        best_model.eval()
        holdout_result = run_evaluation(
            best_model,
            device,
            cfg,
            seed=cfg.final_eval_seed,
            games_count=cfg.final_eval_games,
            label="test finale hold-out del checkpoint BEST",
        )

    total_time = time.perf_counter() - start_time
    print(f"Training terminato | tempo totale: {format_duration(total_time)}", flush=True)
    print(
        f"Punteggio finale ({cfg.eval_mode}, {cfg.eval_games} match): "
        f"{mean:.2f} ± {std:.2f}",
        flush=True,
    )
    print(f"Checkpoint LAST: {cfg.save_path}", flush=True)
    print(
        f"Miglior checkpoint: {best_eval_mean:.2f} ± {best_eval_std:.2f} "
        f"a {best_eval_episodes:,} episodi",
        flush=True,
    )
    print(f"Checkpoint BEST: {cfg.best_save_path}", flush=True)
    if holdout_result is not None:
        holdout_mean, holdout_std = holdout_result
        holdout_sem = holdout_std / math.sqrt(cfg.final_eval_games)
        print(
            f"Risultato hold-out BEST ({cfg.final_eval_games:,} match): "
            f"{holdout_mean:.2f} ± {holdout_std:.2f} | "
            f"errore standard={holdout_sem:.3f}",
            flush=True,
        )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Knister Double-DQN con stato engineered184, rete dueling, "
            "checkpoint BEST/LAST e test finale hold-out"
        )
    )
    parser.add_argument("--episodes", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=1024)
    parser.add_argument("--replay-capacity", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-starts", type=int, default=100_000)
    parser.add_argument("--updates-per-vector-step", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument(
        "--lr-end",
        type=float,
        default=None,
        help=(
            "Se impostato, il learning rate decresce linearmente da --lr a "
            "--lr-end sulla frazione --lr-decay-fraction degli aggiornamenti "
            "gradiente totali stimati, poi resta costante. Default: nessun "
            "decadimento (comportamento identico alla versione precedente)."
        ),
    )
    parser.add_argument(
        "--lr-decay-start-fraction",
        type=float,
        default=0.0,
        help=(
            "Il learning rate resta fisso a --lr fino a questa frazione degli "
            "aggiornamenti gradiente totali stimati, poi decresce fino a "
            "--lr-end alla frazione --lr-decay-fraction. Default 0.0 = "
            "decadimento da subito."
        ),
    )
    parser.add_argument("--lr-decay-fraction", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument(
        "--network-type",
        type=str,
        default="dueling",
        choices=["dueling", "mlp"],
        help="'dueling' separa valore dello stato e vantaggio delle 25 azioni.",
    )
    parser.add_argument(
        "--state-encoding",
        type=str,
        default="engineered192",
        choices=["engineered192", "engineered184", "onehot", "raw"],
        help=(
            "engineered192: 184 feature dense + 8 con la tabella di scoring "
            "corrente (necessario con --randomize-scores); engineered184: le "
            "184 feature originali, senza condizionamento sullo score, utile "
            "come controllo/ablation; onehot: codifica da 312; raw: 26 valori."
        ),
    )
    parser.add_argument("--target-update-every", type=int, default=4_000)
    parser.add_argument(
        "--randomize-scores",
        action="store_true",
        help=(
            "Campiona una tabella di punteggi diversa ad ogni round vettoriale "
            "invece di usare sempre quella di api.KnisterGame (""requisito di "
            "conformità su modifica punteggi senza retraining). "
            "La valutazione usa comunque sempre la tabella vera."
        ),
    )
    parser.add_argument("--score-jitter-low", type=float, default=SCORE_JITTER_LOW)
    parser.add_argument("--score-jitter-high", type=float, default=SCORE_JITTER_HIGH)
    parser.add_argument(
        "--no-dueling-mask-aware",
        action="store_false",
        dest="dueling_mask_aware",
        help=(
            "Disattiva la media mascherata sulle sole azioni valide "
            "nell'aggregazione dueling, riproducendo il comportamento "
            "precedente (utile solo per l'A/B diretto, #4)."
        ),
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=1,
        help="Ampiezza del ritorno n-step (1 = transizioni a singolo step, come prima).",
    )
    parser.add_argument(
        "--include-current-line-scores",
        action="store_true",
        help=(
            "Aggiunge 75 feature con lo score corrente (non proiettato) di "
            "riga/colonna/diagonale per ciascuna azione."
        ),
    )
    parser.add_argument(
        "--prioritized-replay",
        action="store_true",
        help=(
            "Prioritized Experience Replay ("
            "§9.5): campiona più spesso le transizioni con TD-error assoluto "
            "maggiore, con correzione di importance sampling nella loss."
        ),
    )
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--per-epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--symmetry-augmentation",
        action="store_true",
        help=(
            "Data augmentation nel replay con le 8 simmetrie D4 della "
            "griglia: score "
            "invariante per costruzione, verificato in "
            "validate_symmetry_augmentation()."
        ),
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--log-every-episodes",
        type=int,
        default=20_000,
        help="Stampa progresso ogni circa questo numero di episodi (0 = usa i batch).",
    )
    parser.add_argument(
        "--progress-window",
        type=int,
        default=5,
        help="Numero di intervalli recenti usati per stimare velocità ed ETA.",
    )
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--eval-every-episodes", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=9_876)
    parser.add_argument(
        "--final-eval-games",
        type=int,
        default=10_000,
        help="Match del test finale separato sul BEST; 0 lo disattiva.",
    )
    parser.add_argument("--final-eval-seed", type=int, default=24_681_357)
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="vectorized",
        choices=["vectorized", "official"],
        help=(
            "vectorized è molto più veloce; official usa api.KnisterGame "
            "in blocchi ed è utile come controllo finale."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1024,
        help="Numero massimo di match paralleli nella valutazione vettorizzata.",
    )
    parser.add_argument(
        "--eval-progress-every",
        type=int,
        default=50,
        help="In modalità official, completa e stampa la valutazione a blocchi di N match.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="modello_knister_v2_last.pth",
    )
    parser.add_argument(
        "--best-save-path",
        type=str,
        default="modello_knister_v2_best.pth",
    )
    args = parser.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    train(parse_args())
