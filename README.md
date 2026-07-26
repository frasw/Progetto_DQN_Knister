# Progetto Knister — Agente DQN

Agente Deep Q-Network per il gioco Knister (griglia 5×5, 25 mosse, punteggio da
combinazioni poker-style su righe/colonne/diagonali). Risultato finale:
**63,19** di punteggio medio hold-out (architettura LineNet + ritorni a 3 passi),
verificato con l'API ufficiale del docente.

---

## Contenuto e a cosa serve ogni file

| File | Ruolo | Modificabile? |
|---|---|---|
| `api.py` | Il gioco, fornito dal docente | No, mai toccato |
| `train_knister_v2.py` | Motore completo: ambiente vettorizzato, replay, encoder, loop di training Double DQN, valutazione, tutte le leve dello studio di ablazione | Sì (nostro codice) |
| `linenet.py` | Architettura LineNet + encoder dedicato + validatori di correttezza | Sì (nostro codice) |
| `linenet_train.py` | Entrypoint che innesta LineNet nel training di `train_knister_v2.py` senza modificarlo | Sì (nostro codice) |
| `eval_tools.py` | Verifica di un checkpoint (API ufficiale, sanity check, generalizzazione) | Sì (nostro codice) |
| `linenet_n3_3M_seed7_best.pth` (+ `.json`) | **Modello finale** (63,19) | Output |
| `linenet_n3_3M_seed42_best.pth` (+ `.json`) | Modello seed 42 (61,29), conferma di stabilità | Output |

> **Importante:** ogni `.pth` ha accanto un file `.json` con lo stesso nome più
> `.json`. Il `.json` contiene i metadati per ricostruire la rete: **i due file
> vanno sempre tenuti insieme.** Rinominando un `.pth`, rinomina anche il suo
> `.json`.

I quattro file `.py` nostri sono **interdipendenti**: `linenet.py` e
`linenet_train.py` importano funzioni da `train_knister_v2.py`, che a sua volta
usa `api.py`. Vanno tutti nella stessa cartella.

---

## Riprodurre il risultato finale

### 1. Addestrare (opzionale — il checkpoint è già incluso)

Su GPU (es. Google Colab), dopo aver caricato i file `.py` nella stessa cartella:

```bash
python linenet_train.py \
  --episodes 3000000 \
  --n-step 3 \
  --seed 7 \
  --device cuda \
  --final-eval-games 10000 \
  --save-path modello_last.pth \
  --best-save-path modello_best.pth
```

Durata: circa 1 ora e 25 minuti su GPU. Al termine salva due checkpoint
(`_last` e `_best`) con i rispettivi `.json`. Tutti gli altri iperparametri
usano i valori di default già ottimizzati (vedi sotto).

### 2. Verificare un checkpoint con l'API ufficiale del docente

Questo fa giocare il modello **direttamente con `api.KnisterGame`**, non con
l'ambiente veloce interno — è la verifica che il risultato è reale:

```bash
python eval_tools.py official \
  --checkpoint modello_best.pth \
  --games 10000 \
  --device cuda
```

Stampa media e deviazione standard su 10.000 partite. Atteso: ~63.

### 3. Ispezionare un checkpoint (controllo rapido)

```bash
python eval_tools.py info --checkpoint modello_best.pth --device cpu
```

Mostra architettura, numero di parametri e una sanity check veloce su 256
partite.

---

## Iperparametri principali

I default sono già impostati sui valori finali; questi sono i flag che contano
se si vuole sperimentare.

### Flag di `linenet_train.py` (e `train_knister_v2.py`)

| Flag | Default | Significato |
|---|---|---|
| `--episodes` | 3000000 | Numero di partite di training. Più episodi = più apprendimento, con rendimento decrescente. |
| `--n-step` | 3 | Passi di reward reale nel target (credit assignment). **La leva più importante**: da 1 a 3 vale ~+11 punti. |
| `--seed` | 42 | Seed di inizializzazione. Cambiarlo dà un modello diverso ma di prestazioni simili (stabilità verificata). |
| `--device` | cuda | `cuda` per GPU, `cpu` altrimenti (molto più lento). |
| `--final-eval-games` | 5000 | Partite dell'hold-out finale. Più alto = stima più precisa. |
| `--save-path` / `--best-save-path` | — | Dove salvare il checkpoint dell'ultimo episodio / del migliore. |

### Flag specifici di LineNet

| Flag | Default | Significato |
|---|---|---|
| `--line-embed-dim` | 64 | Dimensione dell'embedding per cella. |
| `--line-ctx-dim` | 128 | Dimensione del contesto di linea. |
| `--line-head-dim` | 256 | Dimensione della testa dueling. |
| `--skip-startup-validators` | off | Salta i 4 validatori di correttezza all'avvio (sconsigliato). |

> **Nota:** con `linenet_train.py` **non** vanno passati `--network-type`,
> `--state-encoding` o `--compile`: sono gestiti internamente e danno errore
> esplicito se specificati.

### Iperparametri fissi (raramente da toccare)

`--gamma 1.0` (sconto nullo, corretto per orizzonte fisso a 25 passi),
`--lr 0.00025` (learning rate), `--replay-capacity 2000000`, `--batch-size 1024`,
`--target-update-every 4000`, `--epsilon-start 1.0 --epsilon-end 0.03
--epsilon-decay-fraction 0.50` (esplorazione: parte casuale, diventa greedy
entro metà training). L'epsilon riguarda **solo il training**: la valutazione è
sempre greedy pura.

---

## Nota sull'ambiente veloce

L'addestramento usa `FastVectorKnister`, una reimplementazione vettorizzata del
gioco che esegue 1024 partite in parallelo (necessaria per la scala di milioni
di episodi). La sua equivalenza all'`api.py` ufficiale è:
- **provata** a ogni avvio del training da un validatore automatico;
- **confermata empiricamente** dalla verifica `eval_tools.py official`, che dà lo
  stesso risultato entro 0,04 punti.

Per ogni dubbio sulla fedeltà del risultato, il comando `official` è la fonte di
verità definitiva perché usa il codice del docente senza intermediari.
