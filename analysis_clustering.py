import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# k-MEANS Algorithm 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.interpolate import make_interp_spline



# # Supponiamo che data_elbow sia un dizionario o un DataFrame
# # Creiamo dei dati di esempio simili alla struttura attesa
# # In un caso reale, utilizzeresti i tuoi dati effettivi
# # Esempio di dati per il metodo del gomito
# k_values = list(range(1, 11))  # Da 1 a 10 clusters
# inertia_values = [100, 80, 60, 43, 38, 35, 33, 31, 30, 29]  # Valori di inerzia decrescenti
# optimal_k_elbow = 4  # Il valore K ottimale

# # Per simulare i dati di Dash Mantine, creiamo una lista di dizionari
# data_elbow = [{"K": k, "inertia": inertia} for k, inertia in zip(k_values, inertia_values)]

def create_elbow_silhouette_plot(data, optimal_k, num,title="Elbow Method for K-Means Clustering"):
    """
    Crea un grafico del metodo del gomito utilizzando Matplotlib.
    
    Args:
        data: Lista di dizionari con chiavi 'K' e 'inertia'
        optimal_k: Il valore K ottimale da evidenziare
        title: Titolo del grafico
    """
    # Estrai i dati dalla lista di dizionari
    k_values = [item["K"] for item in data]
    try:
        inertia_values = [item["inertia"] for item in data]
    except:
        inertia_values = [item["silhouette"] for item in data]
    
    # Crea una figura di alta qualità
    plt.figure(num=1,figsize=(10, 6), dpi=100)
    
    # Imposta uno stile moderno
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Crea una curva smooth per il grafico (simile a curveType="natural")
    if len(k_values) > 3:  # Abbiamo bisogno di almeno 4 punti per fare una curva smooth
        # Crea punti intermedi per rendere la curva più smooth
        k_smooth = np.linspace(min(k_values), max(k_values), 300)
        # Crea la spline
        spl = make_interp_spline(k_values, inertia_values, k=3)
        # Valuta la spline nei punti smooth
        inertia_smooth = spl(k_smooth)
        
        # Disegna la linea smooth con gradiente di colore blu-violetto
        plt.plot(k_smooth, inertia_smooth, '-', linewidth=5, color='#4263eb', alpha=0.8)
    else:
        # Se non abbiamo abbastanza punti, disegna una linea normale
        plt.plot(k_values, inertia_values, '-', linewidth=5, color='#4263eb', alpha=0.8)
    
    # Aggiungi i punti dati
    plt.scatter(k_values, inertia_values, s=80, color='#4263eb', zorder=5)
    
    # Aggiungi una linea di riferimento verticale per il K ottimale
    plt.axvline(x=optimal_k, color='#e64980', linestyle='--', linewidth=2, 
                label=f'Optimal K = {optimal_k}')
    
    # Aggiungi un marker e un'etichetta per il K ottimale
    try:
        plt.scatter([optimal_k], [data[optimal_k-1]["inertia"]], s=150, 
                    color='#e64980', zorder=10, edgecolor='white', linewidth=2)
    except:
        plt.scatter([optimal_k], [data[optimal_k-1]["silhouette"]], s=150, 
                    color='#e64980', zorder=10, edgecolor='white', linewidth=2)
    
    try:
        plt.annotate(f'Best Cluster (K={optimal_k})',
                xy=(optimal_k, data[optimal_k-1]["inertia"]),
                xytext=(optimal_k+0.5, data[optimal_k-1]["inertia"]),
                fontsize=12,
                color='#e64980',
                weight='bold',
                arrowprops=dict(arrowstyle="->", color='#e64980', lw=1.5))
    except:
        plt.annotate(f'Best Cluster (K={optimal_k})',
                xy=(optimal_k, data[optimal_k-1]["silhouette"]),
                xytext=(optimal_k+0.5, data[optimal_k-1]["silhouette"]),
                fontsize=12,
                color='#e64980',
                weight='bold',
                arrowprops=dict(arrowstyle="->", color='#e64980', lw=1.5))
    
    # Aggiungi titolo e etichette degli assi
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Number of Clusters (K)', fontsize=14, labelpad=10)
    plt.ylabel('Inertia', fontsize=14, labelpad=10)
    
    # Imposta i limiti e i tick degli assi
    plt.xlim(min(k_values) - 0.5, max(k_values) + 0.5)
    plt.xticks(k_values)
    
    # Migliora lo stile del grafico
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Crea una leggenda personalizzata
    legend_elements = [
        Line2D([0], [0], color='#4263eb', lw=4, label='Inertia'),
        Line2D([0], [0], color='#e64980', linestyle='--', lw=2, label=f'Optimal K = {optimal_k}')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    

    
    # Opzionale: Salva il grafico come file immagine
    # plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')

    return plt



def elbow_silhouette_method(df_, columns_selected, cluster_method_custom, cluster_value:int=None, cluster_method_stat:str="elbow"):
    # Subset df according to the columns selected by the user
    df = df_.loc[:, columns_selected].dropna()

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Silhouette and elbow score
    # Use Silhouette and Elbow Analysis to find the optimal number of clusters
    silhouette_scores = []
    inertia = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0, n_init='auto')
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        inertia.append(kmeans.inertia_)

    # Transform the two lists into a list of dictionaries with rounded inertia values
    data_elbow = []
    for k_value, inertia_value in zip(K, inertia):
        data_elbow.append({
            "K": k_value,
            "inertia": round(inertia_value, 2)
        })

    data_silhouette = []
    for k_value, silhouette_value in zip(K, silhouette_scores):
        data_silhouette.append({
            "K": k_value,
            "silhouette": round(silhouette_value, 2)
        })

    # Find the elbow point (simplified method)
    changes = np.diff(inertia)
    second_derivative = np.diff(changes)
    elbow_index = np.argmax(np.abs(second_derivative)) + 2  # +2 due to double differencing
    optimal_k_elbow = K[elbow_index]

    # Find the max silhouette score
    optimal_k_silhouette = K[np.argmax([round(float(val), 2) for val in silhouette_scores])]+ 2 

    # Create plots
    plt_1 = create_elbow_silhouette_plot(data_elbow, optimal_k_elbow, num=1)
     # Mostra il grafico
    plt_1.show()
    plt_2 = create_elbow_silhouette_plot(data_silhouette, optimal_k_silhouette, num=2, title="Silhouette Method for K-Means Clustering")
    plt_2.show()

    if cluster_method_custom == True:
        optimal_k = cluster_value
    else: 
        if cluster_method_stat == "elbow":
            optimal_k = optimal_k_elbow
        else:
            optimal_k = optimal_k_silhouette
    # Cluster data with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0, n_init='auto')
    kmeans.fit(X_scaled)

    # Plot the clusters and their centers
    centers = kmeans.cluster_centers_
    labels_cluster = kmeans.labels_
    df_['cluster'] = labels_cluster.astype('str')
    
    return optimal_k_silhouette, X_scaled, df_, optimal_k


def pulisci_e_ricrea_file(cartella):
    """
    Controlla se nella cartella specificata ci sono file.
    Se sì, li cancella tutti e poi crea i nuovi file specificati.
    
    Args:
        cartella (str): Percorso della cartella da controllare/pulire
        nuovi_file (list, optional): Lista di nomi dei nuovi file da creare
            Default None crea file di esempio
    """
    # Verifica che la cartella esista
    if not os.path.exists(cartella):
        print(f"La cartella {cartella} non esiste. La sto creando...")
        os.makedirs(cartella)
        print(f"Cartella {cartella} creata con successo.")
    
    # Controlla se ci sono file nella cartella
    contenuto = os.listdir(cartella)
    file_presenti = [f for f in contenuto if os.path.isfile(os.path.join(cartella, f))]
    
    # Se ci sono file, cancellali
    if file_presenti:
        print(f"Trovati {len(file_presenti)} file nella cartella {cartella}:")
        for file in file_presenti:
            file_path = os.path.join(cartella, file)
            try:
                os.remove(file_path)
                print(f"  - Cancellato: {file}")
            except Exception as e:
                print(f"  - Errore nella cancellazione di {file}: {str(e)}")
    else:
        print(f"Nessun file trovato nella cartella {cartella}")

def generate_dataset_cluster(
    df_, 
    columns_selected, 
    cluster_method_custom:bool=False, 
    cluster_value:int=2, 
    cluster_method_stat:str="elbow",
    delete_columns:bool=False,
    column_to_delete = [],
    save_df_cluster:bool=False,
    path_folder_save_df_cluster:str=""):
    
    # Rimuovi data dove average_opaque_surface_transmittance è 0.1
    df=df_.copy()
    df = df[df['average_opaque_surface_transmittance'] > 0.1]
    # Rimuovi data dove average_glazed_surface_transmittance è minore di 0.5
    df = df[df['average_glazed_surface_transmittance'] > 0.5]
    
    _,_,df_cluster, optimal_k = elbow_silhouette_method(df, columns_selected, cluster_method_custom, cluster_value, cluster_method_stat)
    if save_df_cluster:
        pulisci_e_ricrea_file(path_folder_save_df_cluster)
        for cluster in range(optimal_k):
            df_cluster_ = df_cluster[df_cluster['cluster'] == str(cluster)]
            if delete_columns:
                df_cluster_ = df_cluster_.select_dtypes(include=[np.number]).drop(columns=column_to_delete)
            df_cluster_.to_csv(f"{path_folder_save_df_cluster}/cluster_{cluster}.csv", sep=",", decimal=".",index=False)
    
    return df_cluster, optimal_k


