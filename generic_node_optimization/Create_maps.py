import h5py
import pandas as pd
import folium
from folium.plugins import PolyLineOffset, PolyLineTextPath
import branca.colormap as cm
from pathlib import Path
import numpy as np

# === INPUTS ===
h5_path = Path(
    r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\FirstWork simulations\userData\20250915135513-1\optimization_results.h5")
selected_period = "period1"
selected_network = "hydrogenPipelineOnshore_lowP"

# === DEFINIZIONE NODI ===
nodes = ["Roermond", "Dongen", "Oosterhout", "Maastricht", "Venlo", "Arnhem", "East_Groningen",
         "Betuwe", "Heerlen", "Dordrecht", "Wageningen", "Rotterdam", "Zeeland", "North_Sea",
         "North_Netherlands", "Chemelot", "Zuidwending"]

node_lon = {
    'Roermond': 5.9875, 'Dongen': 4.9459, 'Oosterhout': 4.8617, 'Maastricht': 5.6910,
    'Venlo': 6.1670, 'Arnhem': 5.8987, 'East_Groningen': 6.7261, 'Betuwe': 5.5000,
    'Heerlen': 5.9815, 'Dordrecht': 4.6783, 'Wageningen': 5.6654, 'Rotterdam': 4.4777,
    'Zeeland': 3.8497, 'North_Sea': 4.8170, 'North_Netherlands': 6.8262,
    'Chemelot': 5.8004, 'Zuidwending': 6.9332
}

node_lat = {
    'Roermond': 51.1942, 'Dongen': 51.6247, 'Oosterhout': 51.6410, 'Maastricht': 50.8514,
    'Venlo': 51.3670, 'Arnhem': 51.9851, 'East_Groningen': 53.1722, 'Betuwe': 51.9167,
    'Heerlen': 50.8837, 'Dordrecht': 51.7958, 'Wageningen': 51.9692, 'Rotterdam': 51.9244,
    'Zeeland': 51.4988, 'North_Sea': 52.4330, 'North_Netherlands': 53.454370,
    'Chemelot': 50.9756, 'Zuidwending': 53.0955
}


# === FUNZIONE DEBUG PER ESPLORARE IL FILE H5 ===
def explore_h5_structure(h5_path):
    """Esplora la struttura del file H5 per capire i percorsi disponibili"""
    print("=== STRUTTURA FILE H5 ===")
    try:
        with h5py.File(h5_path, "r") as f:
            def print_structure(name, obj):
                print(f"📁 {name}")
                if isinstance(obj, h5py.Dataset):
                    print(f"   📄 Dataset shape: {obj.shape}, dtype: {obj.dtype}")
                    if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                        print(f"   🏷️  Attributes: {dict(obj.attrs)}")

            print("\n--- Struttura completa del file ---")
            f.visititems(print_structure)

            # Verifica percorso specifico
            base_path = f"operation/networks/{selected_period}/{selected_network}"
            print(f"\n--- Verifica percorso: {base_path} ---")
            if base_path in f:
                print(f"✅ Percorso trovato!")
                group = f[base_path]
                print(f"Numero di archi: {len(group.keys())}")
                print(f"Archi disponibili: {list(group.keys())}")
            else:
                print(f"❌ Percorso NON trovato!")
                print("Percorsi disponibili in 'operation/networks':")
                if "operation" in f and "networks" in f["operation"]:
                    for period in f["operation/networks"].keys():
                        print(f"  📅 Periodo: {period}")
                        for network in f[f"operation/networks/{period}"].keys():
                            print(f"    🌐 Network: {network}")
    except Exception as e:
        print(f"❌ Errore nell'aprire il file: {e}")


# === FUNZIONE MIGLIORATA PER LEGGERE FLUSSI ===
def load_flows_debug(h5_path, period, network):
    """Versione debug della funzione per leggere i flussi"""
    flow_data = []

    try:
        with h5py.File(h5_path, "r") as f:
            base_path = f"operation/networks/{period}/{network}"
            print(f"\n=== CARICAMENTO FLUSSI ===")
            print(f"Percorso: {base_path}")

            if base_path not in f:
                print(f"❌ Path {base_path} non trovato nel file H5")
                return pd.DataFrame()

            group = f[base_path]
            print(f"✅ Trovati {len(group.keys())} archi")

            for i, arc_id in enumerate(group.keys()):
                print(f"\n--- Arco {i + 1}/{len(group.keys())}: {arc_id} ---")
                arc_group = f[f"{base_path}/{arc_id}"]

                # Debug attributi
                print(f"Attributi: {dict(arc_group.attrs)}")
                print(f"Dataset disponibili: {list(arc_group.keys())}")

                if "flow" not in arc_group:
                    print("⚠️ Dataset 'flow' non trovato")
                    continue

                flow_dataset = arc_group["flow"]
                print(f"Shape del dataset flow: {flow_dataset.shape}")
                print(f"Dtype: {flow_dataset.dtype}")

                # Leggi e analizza i flussi
                flow_values = flow_dataset[:]
                print(
                    f"Valori flow: min={flow_values.min():.4f}, max={flow_values.max():.4f}, mean={flow_values.mean():.4f}")

                # Prova diverse strategie di aggregazione
                flow_sum = flow_values.sum()
                flow_mean = flow_values.mean()
                flow_max = flow_values.max()

                print(f"Aggregazioni: sum={flow_sum:.4f}, mean={flow_mean:.4f}, max={flow_max:.4f}")

                # Ottieni nodi
                from_node = arc_group.attrs.get("from_node", None)
                to_node = arc_group.attrs.get("to_node", None)

                if from_node is not None:
                    from_node = from_node.decode() if isinstance(from_node, bytes) else str(from_node)
                if to_node is not None:
                    to_node = to_node.decode() if isinstance(to_node, bytes) else str(to_node)

                print(f"Da: {from_node} → A: {to_node}")

                # Usa il valore assoluto del flusso massimo se la somma è troppo piccola
                final_flow = flow_sum if abs(flow_sum) > 1e-6 else abs(flow_max)

                flow_data.append({
                    "Arc_ID": arc_id,
                    "FromNode": from_node,
                    "ToNode": to_node,
                    "Flow": final_flow,
                    "Flow_Sum": flow_sum,
                    "Flow_Mean": flow_mean,
                    "Flow_Max": flow_max
                })

    except Exception as e:
        print(f"❌ Errore nel caricamento: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(flow_data)
    print(f"\n=== RISULTATI FINALI ===")
    print(f"Totale archi caricati: {len(df)}")
    if not df.empty:
        print(f"Flussi > 0: {len(df[df['Flow'] > 0])}")
        print(f"Range flussi: {df['Flow'].min():.4f} - {df['Flow'].max():.4f}")
        print("\nPrimi 5 archi con flusso maggiore:")
        print(df.nlargest(5, 'Flow')[['FromNode', 'ToNode', 'Flow']])

    return df


# === FUNZIONE MIGLIORATA PER PLOTTARE ===
def plot_network_debug(df, node_lat, node_lon):
    """Versione debug della funzione di plotting"""
    print(f"\n=== CREAZIONE MAPPA ===")

    if df.empty:
        print("❌ DataFrame vuoto - nessun flusso da visualizzare")
        return None

    # Filtra solo flussi positivi significativi
    df_filtered = df[df["Flow"] > 1e-6].copy()  # Soglia più bassa
    print(f"Archi con flusso > 1e-6: {len(df_filtered)}")

    if df_filtered.empty:
        print("❌ Nessun flusso significativo trovato")
        # Crea comunque una mappa con i nodi
        map_center = [sum(node_lat.values()) / len(node_lat), sum(node_lon.values()) / len(node_lon)]
        m = folium.Map(location=map_center, zoom_start=7)

        # Plot solo i nodi
        for node in nodes:
            if node in node_lat:
                folium.CircleMarker(
                    location=(node_lat[node], node_lon[node]),
                    radius=8,
                    popup=f"{node}<br>Nessun flusso",
                    color="red",
                    fill=True,
                    fill_opacity=0.8
                ).add_to(m)

        return m

    # Centra la mappa
    map_center = [sum(node_lat.values()) / len(node_lat), sum(node_lon.values()) / len(node_lon)]
    m = folium.Map(location=map_center, zoom_start=7)

    # Colormap con range più ampio
    max_flow = df_filtered["Flow"].max()
    min_flow = df_filtered["Flow"].min()
    print(f"Range flussi per colormap: {min_flow:.6f} - {max_flow:.6f}")

    # Usa scala logaritmica se c'è grande differenza
    if max_flow / min_flow > 1000:
        print("Usando scala logaritmica per i colori")
        df_filtered["Flow_Log"] = np.log10(df_filtered["Flow"] + 1e-9)
        colormap = cm.linear.OrRd_09.scale(df_filtered["Flow_Log"].min(), df_filtered["Flow_Log"].max())
        use_log = True
    else:
        colormap = cm.linear.OrRd_09.scale(min_flow, max_flow)
        use_log = False

    # Plot archi
    archi_plottati = 0
    for _, row in df_filtered.iterrows():
        from_node = row["FromNode"]
        to_node = row["ToNode"]

        if from_node not in node_lat or to_node not in node_lat:
            print(f"⚠️ Nodi non trovati: {from_node} → {to_node}")
            continue

        from_coords = (node_lat[from_node], node_lon[from_node])
        to_coords = (node_lat[to_node], node_lon[to_node])

        flow = row["Flow"]
        color_value = np.log10(flow + 1e-9) if use_log else flow

        # Spessore proporzionale al flusso
        weight = max(2, min(10, 2 + 8 * (flow - min_flow) / (max_flow - min_flow)))

        line = PolyLineOffset(
            [from_coords, to_coords],
            color=colormap(color_value),
            weight=weight,
            opacity=0.8,
            offset=0,
            tooltip=f"<b>{from_node} → {to_node}</b><br>Flow: {flow:.6f}<br>Arc: {row['Arc_ID']}"
        ).add_to(m)

        # Aggiungi frecce direzionali
        PolyLineTextPath(
            line, " ➤ ", repeat=True, offset=7,
            attributes={"font-weight": "bold", "font-size": "14", "fill": "red"}
        ).add_to(m)

        archi_plottati += 1

    print(f"✅ Archi plottati sulla mappa: {archi_plottati}")

    # Plot nodi
    for node in nodes:
        if node in node_lat:
            # Conta flussi in entrata e uscita
            flows_in = df_filtered[df_filtered["ToNode"] == node]["Flow"].sum()
            flows_out = df_filtered[df_filtered["FromNode"] == node]["Flow"].sum()

            folium.CircleMarker(
                location=(node_lat[node], node_lon[node]),
                radius=8,
                popup=f"<b>{node}</b><br>In: {flows_in:.4f}<br>Out: {flows_out:.4f}",
                color="blue",
                fill=True,
                fillColor="lightblue",
                fill_opacity=0.7,
                weight=2
            ).add_to(m)

    # Aggiungi colormap
    colormap.add_to(m)

    # Aggiungi legenda
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 80px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <b>Network Flow Visualization</b><br>
    Total Arcs: {len(df_filtered)}<br>
    Flow Range: {min_flow:.2e} - {max_flow:.2e}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# === MAIN CON DEBUG ===
print("🔍 INIZIO DEBUG")

# 1. Esplora struttura file
explore_h5_structure(h5_path)

# 2. Carica flussi con debug
df_flows = load_flows_debug(h5_path, selected_period, selected_network)
print(f"\n📊 DataFrame finale:")
print(df_flows)

# 3. Crea mappa
mappa = plot_network_debug(df_flows, node_lat, node_lon)
if mappa:
    output_file = "network_map_debug.html"
    mappa.save(output_file)
    print(f"✅ Mappa salvata in {output_file}")
else:
    print("❌ Impossibile creare la mappa")