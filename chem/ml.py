from utils import *

class Cluster:
    """
    It uses the coordinates from the <ChemSpace> object and applies 
    a K-Means algorithm.

    If fp is True, it precomputes the pairwise distances between
    the molecular fingerprints.
    """

    def __init__(self, chemspace, k=None, labels=None, fp=False, plot=False):
        self.cs = chemspace
        self.k = k
        self.labels = labels
        self.fp = fp
        self.plot = plot
        self.clusters = None
        self.real_labels = labels
        self.accuracy = None
        self.RMSE = None



        if self.fp:
            self.coordinates = list()
            # For earch set in the ChemSpace compute the molecular fingerprint AllChem.GetMorganFingerprintAsBitVect(x,2,1024)
            for molecules in self.cs.hash_space:
                for mol in self.cs.hash_space[molecules]:
                    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))
                    self.coordinates.append(fp)
        else:

            if self.cs.coordinates is not None:
                self.coordinates = self.cs.coordinates
            else:
                # Scale the cs.properties and store them in the coordinates
                #scaler = StandardScaler()
                #self.coordinates = scaler.fit_transform(self.cs.properties)
                
                self.coordinates = self.cs.properties
        print("------->")
        print(self.coordinates)
        self.clusters = self.fit()
        self.accuracy = self.accuracy_score()


    def fit(self):
        """
        Apply the K-Means algorithm to the coordinates.
        """
      
        self.clusters = KMeans(n_clusters=self.k, random_state=0).fit_predict(self.coordinates)

        return self.clusters
    

    def accuracy_score(self):
        """
        Compute the accuracy of the clustering.
        """
        return accuracy_score(self.real_labels, self.clusters)

