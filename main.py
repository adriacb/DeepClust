from utils import *
from ann import Autoencoder
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
from chem.Chem import ChemSpace, Molecule, Molecules
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score    
from tqdm import tqdm 
import matplotlib.cm as cm
import umap

def plot_history():
    f2 = open('history.pkl', 'rb+')
    history = pickle.load(f2)   
    plt.figure(figsize=(12,8))
    plt.title("")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Test')
    plt.legend()
    plt.savefig('/home/adria/TFM/images/loss.png')
    plt.show()
    plt.close()

def preprocess_dataframe(df):
    #create ChemSpace 
    csdf = ChemSpace(df)

    return csdf

def preprocess():
    """
    Read Datasets
    Compute descriptors with Molecule, Molecules, ChemSpace objects
    Appply dim red with Encoder
    Apply KMeans, TSNE
    """
    #plot_history()

    #tox21 = pd.read_csv('data/tox21.csv.gz',compression='gzip')
    bbbp = pd.read_csv('data/BBBP.csv')
    hiv = pd.read_csv('data/HIV.csv')
    #print(tox21.head())


    # tox21mols2 = []
    # tox21_labels = []

    # for mol, lab in zip(bbbp['smiles'], bbbp['p_np']):
    #     try:
    #         molec = Chem.MolFromSmiles(mol)
    #         if type(molec) == Chem.rdchem.Mol:
    #             bbbpmols2.append(molec)
    #             bbbp_labels.append(lab)
    #     except:
    #         print("Molecule convert error.")
    
  
    
    # assert all(isinstance(n, Chem.rdchem.Mol) for n in bbbpmols2)

    # tox21mols= Molecules(tox21['smiles'], "Tox21")
    # print(tox21mols)
    # print(tox21mols.properties)
    #tox21_labels = tox21['p_np'].values


    bbbpmols2 = []
    bbbp_labels = []

    for mol, lab in zip(bbbp['smiles'], bbbp['p_np']):
        try:
            molec = Chem.MolFromSmiles(mol)
            if type(molec) == Chem.rdchem.Mol:
                bbbpmols2.append(molec)
                bbbp_labels.append(lab)
        except:
            print("Molecule convert error.")
    
  
    
    assert all(isinstance(n, Chem.rdchem.Mol) for n in bbbpmols2)

  

    bbbpmols = Molecules(bbbpmols2, 'BBBP')
    print(bbbpmols)
    print(bbbpmols.properties)


    #bbbp_labels = bbbp['p_np'].values
    #print(np.unique(bbbp_labels))

    hivmols = Molecules(hiv['smiles'], 'HIV')
    #print(hivmols)
    #print(hivmols.properties)

    hiv_labels = hiv['HIV_active'].values
    #print(np.unique(hiv_labels))
    #print(bbbpmols.to_df())
    with open('data/df_bbbp.pkl', 'wb+') as f:
        pickle.dump(bbbpmols.to_df(), f)
    f.close()
    with open('data/df_hiv.pkl', 'wb+') as f:
        pickle.dump(hivmols.to_df(), f)
    f.close()
    # Prepare inputs for Encoder:

    bbbp_in = prepareInput(bbbpmols.to_df().drop(["ID", "ROMol", "setName"], axis=1))
    hiv_in = prepareInput(hivmols.to_df().drop(["ID", "ROMol", "setName"], axis=1))

    print(bbbp_in.shape)
    ##print(hiv_in.shape)

    # pa
    with open('data/prep_bbbp.pkl', 'wb+') as f:
        pickle.dump(bbbp_in, f)
    f.close()
    with open('data/bbbp_labels.pkl', 'wb+') as f:
        pickle.dump(bbbp_labels, f)
    f.close()



    with open('data/prep_hiv.pkl', 'wb+') as f:
        pickle.dump(hiv_in, f)
    f.close()      
    with open('data/hiv_labels.pkl', 'wb+') as f:
        pickle.dump(hiv_labels, f)
    f.close()



def chooseclusters(X, range_n_clusters):
    for n_clusters in range_n_clusters:
        #print(n_clusters)        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()


def apply_pipelinetsne(input, labels, normalinput):

    
    autoencoder = keras.models.load_model('/home/adria/TFM/models/autoencoder3d.h5')
    encoder = keras.models.load_model('/home/adria/TFM/models/encoder3d.h5')     

    normalinput = normalinput.drop(["ID", "ROMol", "setName"], axis=1)
    predauto = encoder.predict(input)

    #from sklearn.manifold import TSNE
    
    Xtsne = TSNE(n_components=2, perplexity=47.0).fit_transform(predauto)
    #Xtsne = umap.UMAP(n_components=2).fit_transform(predauto)
    
    dftsne = pd.DataFrame(Xtsne)
    dftsne['ytrue'] = labels

    np.random.seed(123)
    from yellowbrick.cluster import KElbowVisualizer
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,40),metric='silhouette', timings= True, locate_elbow=False)
    visualizer.fit(Xtsne)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

    
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,40), timings= True)
    visualizer.fit(Xtsne)        # Fit data to visualizer
    visualizer.show()        # Finalize and render figure

    chooseclusters(Xtsne, [x for x in range(7,15)])

    kmeans = KMeans(n_clusters=9, random_state=0).fit(Xtsne)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(labels, kmeans.labels_)
    pred = kmeans.labels_
    acc = accuracy_score(labels, pred)
    print(acc)

    dftsne['cluster'] = pred
    dftsne.columns = ['x1','x2','cluster', 'ytrue']

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(labels, pred)

    cm_display = ConfusionMatrixDisplay(cm).plot()

    from sklearn.metrics import roc_curve
    from sklearn.metrics import RocCurveDisplay

    fpr, tpr, _ = roc_curve(labels, pred)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    return predauto, kmeans.labels_, acc, pred, kmeans.cluster_centers_, dftsne, kmeans

def plot_predictions(X, labels, pred, centroids):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
    plt.close()

    plt.scatter(X[:, 0], X[:, 1], c=pred, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
    plt.close()   


def plot_predicionstsne(df):

    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    sns.scatterplot(data=df,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[0])
    ax[0].set_title('Visualized on TSNE 2D')
    sns.scatterplot(data=df,x='x1',y='x2',hue='ytrue',legend="full",alpha=0.5,ax=ax[1])
    ax[1].set_title('True labels on TSNE 2D')
    fig.suptitle('Comparing clustering result')
    plt.show()

def tanimmatrix(arr, arr2):
    zeros_distances = np.zeros(arr.shape)
    ones_distances = np.zeros(arr2.shape)


    # intra tanimoto distances
    for mol in tqdm(arr):
        curr_distances = np.array([])
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
        for mol2 in arr:
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2,2,nBits=1024)
            d = DataStructs.FingerprintSimilarity(fp1, fp2)
            curr_distances = np.append(curr_distances, d)
        zeros_distances = np.vstack((zeros_distances, curr_distances))

    # inter tanimoto distances
    for mol in tqdm(arr):
        curr_distances = np.array([])
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
        for mol2 in arr2:
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2,2,nBits=1024)
            d = DataStructs.FingerprintSimilarity(fp1, fp2)
            curr_distances = np.append(curr_distances, d)
        ones_distances = np.vstack((ones_distances, curr_distances))

    ax = sns.heatmap(zeros_distances)
    plt.show()
    plt.close()
    ax2 = sns.heatmap(ones_distances)
    plt.show()
    plt.close()   
    return zeros_distances, ones_distances




def main():
    f2 = open('data/prep_bbbp.pkl', 'rb+')
    f3 = open('data/bbbp_labels.pkl', 'rb+')
    f4 = open('data/prep_hiv.pkl', 'rb+')
    f5 = open('data/hiv_labels.pkl', 'rb+')
    f6 = open('data/df_hiv.pkl', 'rb+')
    f7 = open('data/df_bbbp.pkl', 'rb+')
    
    
    bbbp_in = pickle.load(f2)
    bbbp_labels = pickle.load(f3)
    df_bbbp = pickle.load(f7)
    hiv_in = pickle.load(f4)
    hiv_labels = pickle.load(f5)
    df_hiv = pickle.load(f6)


    # X, kmeanslabels, acc, pred, centroids = apply_pipeline(hiv_in, hiv_labels) # Accuracy: 0.8949595156466554
    # plot_predictions(X, hiv_labels, pred, centroids)
    # print(pred)

    # X, kmeanslabels, acc, pred, centroids = apply_pipeline(bbbp_in, bbbp_labels)
    # plot_predictions(X, bbbp_labels, pred, centroids)
    # print(pred)

    # X, kmeanslabels, acc, pred, centroids, df = apply_pipelinetsne(hiv_in, hiv_labels, df_hiv) # Accuracy: 0.8949595156466554
    # plot_predicionstsne(df)
    # compare_fps(df_hiv, pred)
    # # print(pred)

    X, kmeanslabels, acc, pred, centroids, df, model = apply_pipelinetsne(bbbp_in,  bbbp_labels, df_bbbp)
    plot_predicionstsne(df)
    # #print(pred)
    with open('data/predictions_k9_p18.pkl', 'wb+') as f:
        pickle.dump(pred, f)
    f.close()  
    with open('data/centroids_k9_p18.pkl', 'wb+') as f:
        pickle.dump(centroids, f)
    f.close()
    with open('data/kmeans_k9_p18.pkl', 'wb+') as f:
        pickle.dump(model, f)
    f.close()  

    #compare_fps(df_bbbp, pred)

def tanim_distances(clust1, clust2):
    
    distances = None
    first = True

    for mol in tqdm(clust1):
        curr_distances = np.array([])
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
        for mol2 in clust2:
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2,2,nBits=1024)
            d = DataStructs.FingerprintSimilarity(fp1, fp2)
            curr_distances = np.append(curr_distances, d)
        if first:
            distances = curr_distances
            first = False
        else:
            distances = np.vstack((distances, curr_distances))

    
    return distances

def compute_distances_between_clusters(df, pred):

    labels = np.unique(pred)
    df['labels'] = pred
    df['labels'].astype(float)
    
    mols_clusters = np.array([])
    labels_clusters = np.array([])

    df = df.sort_values('labels')

    for cluster in labels:
        mols = np.array([])
        labs = np.array([])
        for index, row in df.iterrows():
            if row['labels'] == cluster:
                mols = np.append(mols, row['ROMol'])
                labs = np.append(labs, row['labels'])
        mols_clusters = np.append(mols_clusters, mols)
        labels_clusters = np.append(labels_clusters, labs)
    

    distances = tanim_distances(mols_clusters, mols_clusters)
   
    # df_cm = pd.DataFrame(distances, index = pred,
    #                 columns = pred)
    
    # del distances
    # del mols_clusters
    clust_labels = dict()
    for clust in pred:
        if clust not in clust_labels:
            clust_labels[clust] = 0
        else:
            clust_labels[clust] += 1
    
    labelssss = list()
    for key in clust_labels:
        #arr = [" " for x in range(clust_labels[key])]
        arr = [key for x in range(clust_labels[key])]
        #arr[len(arr)//2] = str(key)
        for lab in arr:
            labelssss.append(lab)

    import matplotlib.ticker as ticker
    g = sns.heatmap(distances, xticklabels=clust_labels, yticklabels=clust_labels)
    g.yaxis.set_major_locator(ticker.MultipleLocator(2039//9))
    g.yaxis.set_major_formatter(ticker.ScalarFormatter())
    g.xaxis.set_major_locator(ticker.MultipleLocator(2039//9))
    g.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.title('Tanimoto distances for each cluster')
    plt.savefig('heatmap.png')
    plt.show()
    plt.close()
           
    return distances
    

def main_distances():
    f2 = open('data/prep_bbbp.pkl', 'rb+')
    f3 = open('data/bbbp_labels.pkl', 'rb+')
    f4 = open('data/kmeans_k9_p18.pkl', 'rb+')
    f5 = open('data/predictions_k9_p18.pkl', 'rb+')
    f6 = open('data/centroids_k9_p18.pkl', 'rb+')
    f7 = open('data/df_bbbp.pkl', 'rb+')
    
    
    bbbp_in = pickle.load(f2)
    bbbp_labels = pickle.load(f3)
    df_bbbp = pickle.load(f7)
    model = pickle.load(f4)
    predictions = pickle.load(f5)
    centroids = pickle.load(f6)

    compute_distances_between_clusters(df_bbbp, predictions)

def draw_grid_by_clust():
    f2 = open('data/prep_bbbp.pkl', 'rb+')
    f3 = open('data/bbbp_labels.pkl', 'rb+')
    f4 = open('data/kmeans_k9_p18.pkl', 'rb+')
    f5 = open('data/predictions_k9_p18.pkl', 'rb+')
    f6 = open('data/centroids_k9_p18.pkl', 'rb+')
    f7 = open('data/df_bbbp.pkl', 'rb+')
    bbbp_in = pickle.load(f2)
    bbbp_labels = pickle.load(f3)
    df_bbbp = pickle.load(f7)
    model = pickle.load(f4)
    predictions = pickle.load(f5)
    centroids = pickle.load(f6)

    df_bbbp['labels'] = predictions
    for clust in np.unique(df_bbbp['labels']):
        #print(np.random.choice(df_bbbp[df_bbbp['labels'] == clust]['ROMol'], size=10))
        svg = Chem.Draw.MolsToGridImage(np.random.choice(df_bbbp[df_bbbp['labels'] == clust]['ROMol'], size=10),
        molsPerRow=5, useSVG=False)
        svg.show()

if __name__ == '__main__':
    #preprocess()
    #main()
    #main_distances()
    draw_grid_by_clust()
    # TO DO:
    #   x optimal number of clusters
    #   x Test with morgan fps