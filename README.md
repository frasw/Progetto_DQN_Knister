# Knister — Agente Deep Q-Network

Agente di reinforcement learning per il gioco Knister (griglia 5×5, 25 mosse,
punteggio da combinazioni poker-style su righe, colonne e diagonali).

**Risultato finale: 63,19 di punteggio medio**, verificato con l'API ufficiale
del corso su 10.000 partite hold-out. Il risultato è confermato su due
addestramenti indipendenti (seed 42: 61,29 — seed 7: 63,19).

---

## Indice

- [Requisiti](#requisiti)
- [Struttura del progetto](#struttura-del-progetto)
- [Avvio rapido](#avvio-rapido)
- [`linenet.py` — validatori dell'architettura](#linenetpy--validatori-dellarchitettura)
- [`linenet_train.py` — addestramento](#linenet_trainpy--addestramento)
- [`eval_tools.py` — valutazione dei checkpoint](#eval_toolspy--valutazione-dei-checkpoint)
- [`train_knister_v2.py` — il motore](#train_knister_v2py--il-motore)
- [Iperparametri](#iperparametri)
- [I file dei pesi](#i-file-dei-pesi)

---

## Requisiti

```bash
pip install torch numpy
```

Tutti i comandi funzionano su CPU. Dove è disponibile una GPU è sufficiente
sostituire `--device cpu` con `--device cuda` per un'esecuzione molto più
rapida (necessaria solo per gli addestramenti completi).

---

## Struttura del progetto

| File | Ruolo |
|---|---|
| `api.py` | Implementazione del gioco fornita dal corso. Non modificata. |
| `train_knister_v2.py` | Motore: ambiente vettorizzato, encoder, rete DQN, replay buffer, ciclo di addestramento, valutazione. |
| `linenet.py` | Architettura LineNet, encoder dedicato e validatori di correttezza. |
| `linenet_train.py` | Entrypoint di addestramento per LineNet: innesta l'architettura nel motore senza modificarlo. |
| `eval_tools.py` | Strumenti di valutazione di un checkpoint già addestrato. |
| `Pesi/*.pth` + `*.pth.json` | Modelli addestrati e relativi metadati. |

I quattro file Python del progetto sono interdipendenti e devono trovarsi nella
stessa cartella: `linenet.py` e `linenet_train.py` importano
`train_knister_v2.py`, che a sua volta usa `api.py`.

> **Nota sui checkpoint.** Ogni file `.pth` è accompagnato da un file `.json`
> con lo stesso nome più l'estensione `.json`
> (es. `modello_best.pth` → `modello_best.pth.json`). I due file vanno sempre
> tenuti insieme: il `.json` contiene i metadati necessari a ricostruire la
> rete. Rinominando il `.pth` va rinominato anche il `.json`.

---

## Avvio rapido

Per verificare in pochi secondi che l'ambiente sia configurato correttamente:

```bash
python linenet.py --device cpu
```

Esegue i quattro validatori dell'architettura. Se stampa
`TUTTI I VALIDATORI SUPERATI`, tutto è a posto.

---

## `linenet.py` — validatori dell'architettura

Eseguito direttamente, questo file lancia quattro controlli di correttezza
sull'architettura. Non addestra e non richiede checkpoint.

```bash
python linenet.py --device cpu
```

Su GPU:

```bash
python linenet.py --device cuda
```

Cosa verifica:

| # | Validatore | Cosa controlla |
|---|---|---|
| 1 | Encoding `line` | Forma, tipo e correttezza del vettore di stato a 34 dimensioni |
| 2 | Decomposizione del reward | Che il reward calcolato dalla rete coincida con quello reale dell'ambiente, su 4.800 mosse |
| 3 | Equivarianza D₄ | Che ruotando o riflettendo la griglia i Q-value si trasformino coerentemente, sulle 8 simmetrie |
| 4 | Forma e backward | Che le uscite abbiano dimensioni corrette e i gradienti siano finiti |

Output atteso:

```
[OK] 1/4 encoding 'line': shape, dtype, passthrough, tabelle
[OK] 2/4 decomposizione reward: 4800 mosse verificate, err < 0.001
[OK] 3/4 equivarianza D4 esatta delle Q (8 simmetrie, 2 tabelle, err < 0.0002)
[OK] 4/4 forma/finitezza + backward  |  parametri: 110,786
TUTTI I VALIDATORI SUPERATI
```

---

## `linenet_train.py` — addestramento

### Test rapido (circa 20 secondi su CPU)

Addestramento minimo, utile solo a verificare che l'intera catena funzioni:
addestramento, valutazione periodica, salvataggio e ricarica del checkpoint.
Il punteggio ottenuto non è significativo.

```bash
python linenet_train.py --episodes 512 --n-step 3 --seed 42 --device cpu --n-envs 64 --batch-size 32 --learning-starts 128 --replay-capacity 5000 --updates-per-vector-step 1 --eval-every-episodes 512 --eval-games 20 --final-eval-games 20 --save-path prova_last.pth --best-save-path prova_best.pth
```

### Test intermedio (circa 2 minuti su CPU)

```bash
python linenet_train.py --episodes 1280 --n-step 3 --seed 42 --device cpu --n-envs 64 --batch-size 64 --learning-starts 300 --replay-capacity 10000 --eval-every-episodes 640 --eval-games 20 --final-eval-games 40 --save-path prova_last.pth --best-save-path prova_best.pth
```

### Addestramento completo (circa 1 ora e 25 minuti su GPU)

Riproduce il risultato finale del progetto. Richiede una GPU.

```bash
python linenet_train.py --episodes 3000000 --n-step 3 --seed 7 --device cuda --final-eval-games 10000 --save-path modello_last.pth --best-save-path modello_best.pth
```

Tutti gli iperparametri non specificati usano i valori predefiniti, che sono
quelli della configurazione finale. Al termine vengono salvati due checkpoint,
`_last` (ultimo episodio) e `_best` (miglior punteggio ottenuto), ciascuno con
il proprio file `.json`.

> **Flag da non usare con `linenet_train.py`:** `--network-type`,
> `--state-encoding` e `--compile` sono gestiti internamente e il comando
> fallisce con un errore esplicito se vengono passati.

---

## `eval_tools.py` — valutazione dei checkpoint

Quattro sottocomandi, tutti operanti su un checkpoint già addestrato.

### `info` — ispezione rapida

Carica il checkpoint, ricostruisce la rete, stampa i metadati ed esegue una
valutazione di controllo su 256 partite. È il modo più veloce per verificare
che un checkpoint sia integro e caricabile.

```bash
python eval_tools.py info --checkpoint linenet_n3_3M_seed7_best.pth --device cpu
```

### `official` — verifica con l'API ufficiale

**È il comando più importante.** Fa giocare il modello direttamente con la
classe `KnisterGame` di `api.py`, senza passare dall'ambiente vettorizzato
interno: la rete si limita a scegliere dove piazzare il dado, mentre generazione
dei dadi, aggiornamento della griglia e calcolo del punteggio sono interamente
gestiti dal codice del corso.

```bash
python eval_tools.py official --checkpoint linenet_n3_3M_seed7_best.pth --games 10000 --seed 24681357 --device cpu
```

Risultato atteso: circa **63,2**.

Per una verifica più rapida (circa un minuto su CPU):

```bash
python eval_tools.py official --checkpoint linenet_n3_3M_seed7_best.pth --games 1000 --seed 555001 --device cpu
```

Sull'altro checkpoint disponibile, per confermare la riproducibilità del
risultato su un addestramento indipendente (atteso: circa **61,3**):

```bash
python eval_tools.py official --checkpoint linenet_n3_3M_seed42_best.pth --games 10000 --seed 24681357 --device cpu
```

### `lookahead` — valutazione con ricerca a un passo

Anziché scegliere l'azione con il Q-value massimo, considera per ogni mossa
tutti gli esiti possibili del lancio successivo, pesati con la loro probabilità
esatta. Confronta il risultato con la politica greedy sugli stessi semi e
riporta la percentuale di accordo tra le due.

```bash
python eval_tools.py lookahead --checkpoint linenet_n3_3M_seed7_best.pth --games 50 --seed 13579 --batch-games 50 --device cpu
```

> **Attenzione ai tempi.** Per ogni mossa vengono valutati 25×11 stati
> successori, quindi questo comando è molto più lento degli altri: su CPU
> occorrono circa 3 minuti per 50 partite. Su GPU (`--device cuda`) è
> praticabile un numero di partite molto maggiore.

### `zeroshot` — test di generalizzazione

Valuta il modello con tabelle dei punteggi diverse da quella su cui è stato
addestrato, per misurare quanto la strategia appresa dipenda dai valori
specifici delle combinazioni.

```bash
python eval_tools.py zeroshot --checkpoint linenet_n3_3M_seed7_best.pth --games 200 --n-tables 2 --seed 13579 --device cpu
```

Per ogni tabella campionata riporta due valori: `aware`, quando la rete riceve
in input la nuova tabella, e `frozen`, quando l'ambiente usa la nuova tabella
ma la rete continua a vedere quella originale. Richiede pochi secondi su CPU.

### Argomenti comuni

| Argomento | Predefinito | Significato |
|---|---|---|
| `--checkpoint` | obbligatorio | Percorso del file `.pth` |
| `--device` | `auto` | `cpu` oppure `cuda` |
| `--games` | dipende dal comando | Numero di partite da giocare |
| `--seed` | `13579` | Seme di valutazione |
| `--batch-games` | dipende dal comando | Partite valutate in parallelo |

Il seme predefinito (13579) è diverso da quelli usati durante l'addestramento
(9876 per le valutazioni periodiche, 24681357 per l'hold-out finale), così da
non riutilizzare partite già viste in fase di selezione del modello.

---

## `train_knister_v2.py` — il motore

Contiene l'ambiente vettorizzato, il ciclo di addestramento e le procedure di
valutazione. Può essere eseguito direttamente per addestrare l'architettura MLP
originale, usata come riferimento nei confronti:

```bash
python train_knister_v2.py --episodes 512 --n-step 3 --seed 42 --device cpu --n-envs 64 --batch-size 32 --learning-starts 128 --replay-capacity 5000 --state-encoding engineered184 --network-type dueling --eval-games 20 --final-eval-games 20 --save-path mlp_prova_last.pth --best-save-path mlp_prova_best.pth
```

Per l'elenco completo delle opzioni disponibili:

```bash
python train_knister_v2.py --help
```

---

## Iperparametri

I valori predefiniti corrispondono alla configurazione finale: per riprodurre
il risultato non è necessario specificarli.

### Principali

| Flag | Predefinito | Significato |
|---|---|---|
| `--episodes` | 3000000 | Numero di partite di addestramento |
| `--n-step` | 3 | Passi di reward reale nel target. È l'iperparametro con l'impatto maggiore sul risultato |
| `--seed` | 42 | Seme di inizializzazione |
| `--device` | `cuda` | `cuda` oppure `cpu` |
| `--final-eval-games` | 5000 | Partite della valutazione hold-out finale |
| `--save-path` | — | Percorso del checkpoint dell'ultimo episodio |
| `--best-save-path` | — | Percorso del checkpoint migliore |

### Specifici di LineNet

| Flag | Predefinito | Significato |
|---|---|---|
| `--line-embed-dim` | 64 | Dimensione dell'embedding di linea |
| `--line-ctx-dim` | 128 | Dimensione del contesto globale |
| `--line-head-dim` | 256 | Dimensione della testa advantage |
| `--skip-startup-validators` | disattivo | Salta i quattro validatori all'avvio |

Con i valori predefiniti la rete ha 110.786 parametri.

### Valori fissi

`--gamma 1.0` (nessuno sconto: l'orizzonte è fisso a 25 mosse, quindi il ritorno
coincide con il punteggio finale), `--lr 0.00025`,
`--replay-capacity 2000000`, `--batch-size 1024`, `--target-update-every 4000`,
`--epsilon-start 1.0 --epsilon-end 0.03 --epsilon-decay-fraction 0.50`.

L'esplorazione (epsilon) riguarda esclusivamente l'addestramento: tutte le
valutazioni usano una politica greedy pura, senza mosse casuali.

---

## I file dei pesi

| Checkpoint | Punteggio | Note |
|---|---|---|
| `linenet_n3_3M_seed7_best.pth` | **63,19** | Modello finale |
| `linenet_n3_3M_seed42_best.pth` | 61,29 | Addestramento indipendente, conferma la riproducibilità |

Il file `.pth` contiene esclusivamente i pesi appresi (state dict di PyTorch),
non l'architettura né il codice. Il file `.json` affiancato contiene i metadati
necessari a ricostruire la rete corretta in cui caricarli.

I checkpoint `_best` corrispondono al miglior punteggio registrato durante
l'addestramento, che non coincide necessariamente con l'ultimo episodio: nel
modello finale il migliore è stato raggiunto all'episodio 2.900.992 su
3.000.000. Per l'utilizzo va sempre preferito il checkpoint `_best`.

I checkpoint non contengono lo stato dell'ottimizzatore, del replay buffer o dei
generatori di numeri casuali: un addestramento interrotto non può quindi essere
ripreso, ma va rieseguito. È una scelta deliberata, poiché salvare il replay
buffer avrebbe richiesto diversi gigabyte per checkpoint.
