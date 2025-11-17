# Parametric Study Framework

Sistema completo per eseguire studi parametrici multipli su ottimizzazioni di rete energetica.

## 📁 Struttura del Progetto

```
generic_node_optimization/
├── run_parametric_study.py          # Script principale per avviare lo studio
├── optimization_runner.py           # Gestisce singole run di ottimizzazione
├── data_generator.py                # Genera dati Excel basati su parametri
├── analyze_parametric_results.py   # Analisi e visualizzazione risultati
├── Optimize_generic.py              # Script base per singola ottimizzazione
├── Change_data_excel.py             # Utility per modificare dati Excel
├── define_topology.py               # Definisce topologia rete
├── define_components_spec.py        # Specifica componenti
└── results/                         # Cartella risultati
    └── parametric_study_YYYYMMDD_HHMMSS/
        ├── param_config.json        # Configurazione parametri
        ├── results_summary.csv      # Summary di tutti i risultati
        ├── results_summary.xlsx     # Summary in Excel
        ├── SUMMARY.txt              # Riepilogo testuale
        ├── run_0001/                # Risultati run singola
        │   ├── run_params.json      # Parametri della run
        │   ├── run_summary.txt      # Summary della run
        │   ├── input_data/          # Dati input generati
        │   └── model_results/       # Output del modello
        ├── run_0002/
        └── ...
```

## 🎯 Parametri dello Studio

### 1. **DEMAND LEVELS** (`demand_level_ratio`)
- **Descrizione**: Ratio tra domanda nei big clusters vs small clusters
- **Formula**: Domanda_Big / Domanda_Small
- **Valori**: `[3, 5, 7, 10, 15, 20]`
- **Note**: 
  - 2 BIG nodes (BIG1 ha 2x BIG2 in ratio 2:1)
  - 4 SMALL nodes (equamente distribuiti)

### 2. **TOTAL DEMAND** (`total_demand_TWh`)
- **Descrizione**: Domanda totale annuale di idrogeno
- **Unità**: TWh
- **Valori**: `[10, 20, 30, 50, 100]`
- **Note**: Distribuita secondo `demand_level_ratio`

### 3. **IMPORT AVAILABILITY** (`import_availability_ratio`)
- **Descrizione**: Capacità di importazione H2 rispetto alla domanda totale
- **Formula**: Import_Capacity / Total_Demand
- **Valori**: `[0.2, 0.5, 0.7, 1.0]`
- **Note**: 
  - 0.2 = Import limitato (20% della domanda)
  - 1.0 = Import può coprire tutta la domanda

### 4. **IMPORT COST** (`import_cost_multiplier`)
- **Descrizione**: Costo importazione H2 relativo all'elettricità
- **Formula**: H2_price = (Elec_price / 0.65) × multiplier
- **Valori**: `[0.5, 1, 1.5, 2, 3, 5]`
- **Note**: 
  - Base: considera efficienza 65% (1 MWh elec → 0.65 MWh H2)
  - multiplier = 1: H2 import costa come produzione elettrolitica
  - multiplier < 1: H2 import conveniente
  - multiplier > 1: H2 import costoso

### 5. **ELECTRICITY PRICE** (`electricity_price_avg`)
- **Descrizione**: Prezzo medio dell'elettricità
- **Unità**: EUR/MWh
- **Valori**: `[50, 100, 150, 200]`
- **Note**:
  - BIG nodes: prezzo fisso
  - SMALL nodes: variazione ±15% con pattern giornaliero/stagionale

### 6. **ELECTRICITY AVAILABILITY** (`electricity_availability_small`)
- **Descrizione**: Limite di importazione elettricità per SMALL clusters
- **Unità**: MW
- **Valori**: `[30, 50, 100, 200, 400]`
- **Note**: BIG nodes hanno sempre 2000 MW disponibili

### 7. **Scenari Geografici** (`scenarios`)
- **Valori**: `["1751", "1752", "1753"]`
- **Descrizione**: Diverse configurazioni geografiche dei nodi

### 8. **Networks** (`networks`)
- **Opzioni**:
  - `["hydrogenPipelineOnshore_lowP", "hydrogenPipelineOnshore_highP"]`: Entrambe le pressioni
  - `["hydrogenPipelineOnshore_highP"]`: Solo alta pressione

## 🚀 Come Eseguire lo Studio

### 1. Preparazione

Assicurati di avere:
- Python 3.8+
- AdOpT-NET0 installato
- File scenario in `input_data/scenarios/`
- Tutti i moduli richiesti

### 2. Esecuzione

```bash
python run_parametric_study.py
```

Lo script:
1. Mostra il numero totale di combinazioni
2. Chiede conferma prima di iniziare
3. Esegue tutte le ottimizzazioni
4. Salva risultati progressivamente

### 3. Monitoraggio

Durante l'esecuzione vedrai:
```
================================================================================
RUN 45/1000
================================================================================
Scenario: 1751
Total Demand: 30 TWh
Demand Ratio: 7
Import Avail: 0.5
Import Cost Mult: 1.5
Elec Price: 100 EUR/MWh
Elec Avail (small): 100 MW
================================================================================

🚀 Avvio ottimizzazione: run_0045
   📁 Cartella run: results/parametric_study_20250116_143022/run_0045
   💰 H2 Import Price: 230.77 EUR/MWh
   📦 H2 Import Limit: 195.00 MW
   📊 Generazione dati Excel...
   📥 Caricamento carrier data...
   🔧 Costruzione e risoluzione modello...
   ✅ Ottimizzazione completata!
   📈 Objective: 123456789.0
   ⏱️  Time: 85.3s
```

### 4. Interruzione e Ripresa

I risultati sono salvati progressivamente in `results_summary_partial.csv`. Se lo studio viene interrotto, puoi:
- Recuperare i risultati parziali
- Modificare i parametri per continuare da dove ti sei fermato

## 📊 Analisi dei Risultati

### Esegui l'analisi automatica:

```bash
python analyze_parametric_results.py
```

L'analisi produce:
- **Sensitivity plots**: Grafici che mostrano come varia l'objective per ogni parametro
- **Heatmaps**: Mappe di calore per combinazioni di parametri
- **Best configurations**: Top 10 configurazioni migliori
- **Statistical summaries**: Medie, std, min, max per ogni parametro

### File Output

```
results/parametric_study_YYYYMMDD_HHMMSS/analysis/
├── analysis_by_demand_level_ratio.xlsx
├── analysis_by_total_demand_TWh.xlsx
├── analysis_by_import_availability_ratio.xlsx
├── sensitivity_demand_level_ratio.png
├── sensitivity_total_demand_TWh.png
├── heatmap_demand_level_ratio_vs_total_demand_TWh_objective_value.png
├── heatmap_import_availability_ratio_vs_import_cost_multiplier_objective_value.png
└── best_configurations.xlsx
```

## 🔍 Struttura dei Risultati

### results_summary.csv

Contiene una riga per ogni run con:
- Tutti i parametri input
- Parametri derivati (H2 import price, limits, ecc.)
- Risultati del solver (objective, solve time, gap, status)

Esempio:
```csv
run_id;status;scenarios;demand_level_ratio;total_demand_TWh;import_availability_ratio;import_cost_multiplier;electricity_price_avg;electricity_availability_small;objective_value;solve_time;gap
run_0001;SUCCESS;1751;3;10;0.2;0.5;50;30;123456789.0;85.3;0.01
run_0002;SUCCESS;1751;3;10;0.2;0.5;50;50;125678901.0;92.1;0.01
...
```

### run_XXXX/

Ogni cartella run contiene:
- `run_params.json`: Parametri completi della run
- `run_summary.txt`: Summary testuale dettagliato
- `input_data/`: Tutti i file input generati (Excel, JSON, CSV)
- `model_results/`: Output completo del modello AdOpT-NET0

## 🛠️ Personalizzazione

### Modificare la Griglia Parametrica

Edita `run_parametric_study.py`:

```python
param_grid = {
    "demand_level_ratio": [5, 10, 15],  # Riduci il numero di valori
    "total_demand_TWh": [20, 50],       # Focalizza su scenari specifici
    # ... altri parametri
}
```

### Aggiungere Nuovi Parametri

1. Aggiungi al `param_grid` in `run_parametric_study.py`
2. Modifica `data_generator.py` per utilizzare il nuovo parametro
3. Aggiorna `optimization_runner.py` se necessario

### Modificare la Generazione Dati

Edita `data_generator.py` per cambiare:
- Pattern di domanda temporale
- Profili di prezzo elettricità
- Distribuzioni di probabilità

## 📈 Metriche di Performance

Lo studio traccia:
- **Objective Value**: Funzione obiettivo del modello
- **Solve Time**: Tempo di risoluzione in secondi
- **Gap**: MIP gap finale
- **Termination Condition**: Stato finale del solver

## ⚠️ Note Importanti

1. **Spazio Disco**: Ogni run può occupare 50-200 MB. Uno studio completo può richiedere decine di GB.

2. **Tempo di Calcolo**: 
   - Tipicamente 50-200 secondi per run
   - Studio completo (migliaia di run) può richiedere giorni

3. **Memoria**: Le ottimizzazioni sono eseguite sequenzialmente per evitare problemi di memoria.

4. **Sampling**: Per griglie molto grandi, lo script campiona automaticamente un sottoinsieme casuale.

## 🐛 Troubleshooting

### Run Fallite

Se alcune run falliscono:
- Controlla `run_XXXX/run_summary.txt` per errori
- Verifica i parametri in `run_params.json`
- Log di errore salvato in `results_summary.csv` colonna "error"

### Out of Memory

Riduci:
- `threads` nei parametri solver
- Numero di combinazioni nella griglia
- `time_limit` per run lunghe

### File Non Trovati

Verifica:
- Scenario files in `input_data/scenarios/`
- AdOpT-NET0 database accessibile
- Percorsi corretti in `optimization_runner.py`

## 📚 References

- AdOpT-NET0 Documentation
- `Optimize_generic.py` per vedere esempio di singola run
- `define_topology.py` e `define_components_spec.py` per setup componenti

## 👥 Autori

Matteo M. - Gazzani Group

