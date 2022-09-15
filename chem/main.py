
from utils import *
from Molecules import Molecules
from ChemSpace import ChemSpace
from ml_tools import Cluster

def main():
    st = time.time()
    iptacopan = Chem.MolFromSmiles("CCOC1CCN(C(C1)C2=CC=C(C=C2)C(=O)O)CC3=C(C=C(C4=C3C=CN4)C)OC")
    ms_iptacopan = Molecules([iptacopan], "IPTACOPAN")

    iptacopan_s = Chem.SDMolSupplier("/chemotargets/research/SITALA/IPTACOPAN/PCA_Analysis/Distances/iptacopan_scaffold.sdf")
    ms_iptacopan_s = Molecules(iptacopan_s, 'Iptacopan_scaffold')

    enamine_all = Chem.SDMolSupplier("/chemotargets/research/SITALA/IPTACOPAN/PCA_Analysis/Distances/enamine_all_filtered.sdf")
    ms_enamine_all = Molecules(enamine_all, 'Enamine')

    enamine_in = Chem.SDMolSupplier("/chemotargets/research/SITALA/IPTACOPAN/PCA_Analysis/Distances/enamine_filtered.sdf")
    ms_enamine_in = Molecules(enamine_in, 'EnamineIn')

    CFB = Chem.SDMolSupplier("/chemotargets/research/SITALA/IPTACOPAN/PCA_Analysis/Distances/CFB_filtered.sdf")
    ms_CFB = Molecules(CFB, 'CFB')

    InSpace = Chem.SDMolSupplier("/chemotargets/research/SITALA/IPTACOPAN/PCA_Analysis/Distances/InSpace_filtered.sdf")
    ms_InSpace = Molecules(InSpace, 'InSpace')

    # Create a TARGET ChemSpace object
    mw = np.arange(150, 450, 1)
    logP = np.arange(0, 3, 0.1)
    nrings = np.arange(0, 4, 1)
    pfi = np.arange(0, 5, 0.1)
    hbd = np.arange(0,3,1)
    
    region = np.array(np.meshgrid(logP, mw, hbd, nrings, pfi)).T.reshape(-1, 5)


    ms_region = Molecules(region)
    #print(ms_region.properties)


    # CREATE CHEMSPACE OBJECT CONTAINING ALL SETS OF MOLECULES
    cs = ChemSpace([ms_iptacopan, ms_iptacopan_s, ms_enamine_all, ms_enamine_in, ms_CFB, ms_InSpace])
    print(cs, end='\n\n')

    coords, model, scaler = cs.PCA(nComponents=2)

    results = cs.to_df()
    print(results.head())

  
    

    # Select the target region
    region = results[results.setName == ms_region.setName]
    # Select the remaining sets
    dfcopy = results[results.setName != ms_region.setName]


    # # IPTACOPAN distance
    # ipta = results[results.setName == 'IPTACOPAN']
    # d = compute_distance(ipta[['PC1', 'PC2']].to_numpy(), region[['PC1', 'PC2']].to_numpy())
    # print(d)
    

    # for set in dfcopy['setName'].unique():
    #     print(f"Current set: {set}")
    #     curr_group = dfcopy[dfcopy.setName == set]

    #     # Calculate the Euclidean distance between each molecule in the chemspace using compute_distance
    #     distances = compute_distance(curr_group[['PC1', 'PC2']].to_numpy(), region[['PC1', 'PC2']].to_numpy())
    #     results.loc[results.setName == set, 'EuclDist'] = distances






    print(cs.set_names)
    clust = Cluster(cs.properties, k=6, labels=cs.set_names,  plot=True)
    labels = clust.clusters
    print(labels)
    print(clust.real_labels)

    print(clust.RMSE())

    print(f"Accuracy: {clust.accuracy()}")














    elapsed_time = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Plot the distances between the all molecules in the chemspace
    # and the "target" region of the chemspace.
    #plot_distances(results, ref_region=ms_region.setName) #filename='/chemotargets/research/SITALA/IPTACOPAN/PCA_Analysis/Distances/distances.png')
    

def main2():
    bbbp_df= pd.read_csv('/home/adria/Downloads/BBBP_df_revised.csv')
    st = time.time()

    iptacopan_s = Chem.SDMolSupplier("/chemotargets/research/SITALA/IPTACOPAN/PCA_Analysis/Distances/iptacopan_scaffold.sdf")
    ms_iptacopan_s = Molecules(iptacopan_s, 'Iptacopan_scaffold')



    mols_bbbp = Molecules([Chem.MolFromSmiles(x) for x in bbbp_df['smiles']], 'BBBP')

    # Create a ChemSpace object
    cs = ChemSpace([mols_bbbp])
    
    labs = bbbp_df['p_np']
    print(cs, end='\n\n')

    labs_ipta = [3 for x in range(len(iptacopan_s))]

    # add the ipta labels to the labs
    #labs = np.append(labs.to_numpy(), labs_ipta)

    #coords, model, scaler = cs.PCA(nComponents=4)

    clust = Cluster(cs, k=2, labels=labs.to_numpy(),  plot=True, fp=False)


    print(f"Accuracy: {clust.accuracy_score()}")










    elapsed_time = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))



if __name__ == "__main__":
    main2()
