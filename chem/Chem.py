from utils import *
import logging

logger = Logger(filename='molecule.warnings.log', level='WARNING')

class Molecule:
    
    def __init__(self, molecule: Chem.rdchem.Mol):
        self.molecule = molecule
        self.properties = self.calcProp()
        self.props = self.calcProperties()

        if type(molecule) == str:
            self.molecule = Chem.MolFromSmiles(molecule)
    
        try:
            self.name = molecule.GetProp("_Name")
        except:
            logger.warning("Can't get _Name property from Chem.rdchem.Mol")
        else:
            self.name = str(uuid.uuid5())


    def calcProp(self):
        """
        Calculate properties of the molecule
        """
        self.logP = Descriptors.MolLogP(self.molecule)
        self.MWt = Descriptors.MolWt(self.molecule)
        self.nHBD = Descriptors.NumHDonors(self.molecule)
        self.NumAromaticRings = Chem.Lipinski.NumAromaticRings(self.molecule)
        self.PFI = self.logP + self.NumAromaticRings
        self.properties = np.asarray([self.logP, self.MWt, self.nHBD, self.NumAromaticRings, self.PFI])
        
        return self.properties
    
    def calcProperties(self):
        """
        Calculate properties of the molecule
        """
        self.logP = Descriptors.MolLogP(self.molecule)
        self.MWt = Descriptors.MolWt(self.molecule)
        self.nHBD = Descriptors.NumHDonors(self.molecule)
        self.NumAromaticRings = Chem.Lipinski.NumAromaticRings(self.molecule)
        self.PFI = self.logP + self.NumAromaticRings
        self.fracCSP3 = Chem.Lipinski.FractionCSP3(self.molecule)
        self.NumHAcceptors = Chem.Lipinski.NumHAcceptors(self.molecule)
        self.NumHeteroatoms = Chem.Lipinski.NumHeteroatoms(self.molecule)
        self.NumRotatableBonds = Chem.Lipinski.NumRotatableBonds(self.molecule)
        self.NumSaturatedRings = Chem.Lipinski.NumSaturatedRings(self.molecule)
        self.NumAromaticHeterocycles = Chem.Lipinski.NumAromaticHeterocycles(self.molecule)
        self.NumSaturatedHeterocycles = Chem.Lipinski.NumSaturatedHeterocycles(self.molecule)
        self.NumAliphaticHeterocycles = Chem.Lipinski.NumAliphaticHeterocycles(self.molecule)
        self.NumAliphaticRings = Chem.Lipinski.NumAliphaticRings(self.molecule)
        self.NHOHCount = Chem.Lipinski.NHOHCount(self.molecule)
        self.NOCount = Chem.Lipinski.NOCount(self.molecule)
        self.NumRadicalElectrons = Descriptors.NumRadicalElectrons(self.molecule)
        self.NumValenceElectrons = Descriptors.NumValenceElectrons(self.molecule)
        self.NumAliphaticCarbocycles = Chem.Lipinski.NumAliphaticCarbocycles(self.molecule)
        self.NumAromaticCarbocycles = Chem.Lipinski.NumAromaticCarbocycles(self.molecule)
        self.NumSaturatedCarbocycles = Chem.Lipinski.NumSaturatedCarbocycles(self.molecule)
        self.RingCount = Chem.Lipinski.RingCount(self.molecule)
        self.props = np.asarray([self.logP, self.MWt, self.nHBD, self.NumAromaticRings, self.PFI,
                                     self.fracCSP3, self.NumHAcceptors, self.NumHeteroatoms,
                                     self.NumRotatableBonds, self.NumSaturatedRings, #self.NumHeterocycles,
                                     self.NumAromaticHeterocycles, self.NumSaturatedHeterocycles, self.NumAliphaticHeterocycles,
                                     self.NumAliphaticRings, self.NHOHCount, self.NOCount, self.NumRadicalElectrons,
                                     self.NumValenceElectrons, 
                                     self.NumAliphaticCarbocycles, self.NumAromaticCarbocycles, self.NumSaturatedCarbocycles,
                                     self.RingCount])
        
        return self.props
       



class Molecules:
    def __init__(self, molecules, setname=None, infile=None):
        self.setName = str(uuid.uuid1()) if setname is None else setname
        self.molecules = molecules
        
        self.properties = None
        

        
        # It will store the results of the Dimensionality Reduction
        self.coordinates = None
        self.num_molecules = len(molecules)
        if type(molecules) == Chem.rdmolfiles.SDMolSupplier or all(isinstance(n, Chem.rdchem.Mol) for n in molecules):
            self.properties = self._calculate_properties2()
        elif type(molecules) == pd.Series:
            self.molecules = [Chem.MolFromSmiles(x) for x in molecules]
            self.properties = self._calculate_properties2()
        else:
            self.molecules = [f"Molecule_{i+1}" for i in range(len(self.molecules))]
            self.properties = molecules
        
        # It will store the results of the Dimensionality Reduction
        self.coordinates = None
        self.num_molecules = len(molecules)

    
    def _calculate_properties(self):
        
        properties = list()

        for mol in self.molecules:
            logp = Descriptors.MolLogP(mol)
            mwt = Descriptors.MolWt(mol)
            nhbd = Descriptors.NumHDonors(mol)
            nar = Chem.Lipinski.NumAromaticRings(mol)
            pfi = logp + nar
            ps = np.asarray([logp, mwt, nhbd, nar, pfi])

            mol.SetProp("MWt", str(mwt))
            mol.SetProp("logP", str(logp))
            mol.SetProp("nHBD", str(nhbd))
            mol.SetProp("NumAromaticRings", str(nar))
            mol.SetProp("PFI", str(pfi))
            
            if properties is None:
                properties = ps
                print(properties)
            else:
                properties.append(ps)

        return np.array([x for x in properties])

    def _calculate_properties2(self):
        
        properties = list()

        for mol in self.molecules:
            if type(mol) == Chem.rdchem.Mol:
                m = Molecule(mol)
                properties.append(m.props)

        return np.array([x for x in properties])
    
    def to_df(self):
        """
        Convert the Molecules object into a Pandas DataFrame
        ----------
        Returns:
            Pandas DataFrame
        """
        columns = ["ID", "ROMol", "setName", "logP", "MWt", "nHBD", "NumAromaticRings", "PFI"]
        # Add the coordinates if they are available
        if self.coordinates is not None:
            for i in range(self.coordinates.shape[1]):
                columns.append("PC" + str(i+1))

        df = pd.DataFrame(columns=columns)
        # Add the properties
        df["ID"] = [str(i)+"_"+self.setName for i in range(1, self.num_molecules+1)]
        df["ROMol"] = self.molecules
        df["setName"] = [self.setName for i in range(self.num_molecules)]

        desc = [ 'logP',  'MWt',  'nHBD',  'NumAromaticRings',  'PFI',
                                      'fracCSP3',  'NumHAcceptors',  'NumHeteroatoms',
                                      'NumRotatableBonds',  'NumSaturatedRings', 
                                      'NumAromaticHeterocycles',  'NumSaturatedHeterocycles',  'NumAliphaticHeterocycles',
                                      'NumAliphaticRings',  'NHOHCount',  'NOCount',  'NumRadicalElectrons',
                                      'NumValenceElectrons',  
                                      'NumAliphaticCarbocycles',  'NumAromaticCarbocycles',  'NumSaturatedCarbocycles',
                                      'RingCount']

        for i, col in enumerate(desc):
            df[col] = self.properties[:, i]

        # Add the coordinates if they are available
        if self.coordinates is not None:
            for i in range(self.coordinates.shape[1]):
                df["PC" + str(i+1)] = self.coordinates[:, i]

        return df
    

    def to_sdf(self, path=None):
        """
        Convert the Molecules to sdf file.
        ----------
        path: str, optional
            The path to save the sdf file. If not provided, the file will be saved in the current directory.
        """
        if path is None:
            path = os.getcwd()
        
        dataframe = self.to_df()
        
        Chem.PandasTools.WriteSDF(df= dataframe, out = path + "/" + self.setName + ".sdf", 
                                  molColName='ROMol', properties=list(dataframe.columns))
        print("SDF file saved in {}/{}.sdf".format(path, self.setName))
        
        


    def __add__(self):
        """
        TO DO:
        - Add two or more sets of molecules and transform them into a ChemSpace object
        """
        pass

    def __str__(self):
        return "<Molecules> object with {} molecules.\nMolecules set name {}\nProperties dimension: {}".format(len(self.molecules), self.setName, self.properties.shape)


    def __iter__(self):
        return iter(self.molecules)


class ChemSpace:
    """
    Parameters
        chemspace - list of <Molecules> objects
    ===========================
    """

    def __init__(self, chemspace: list[Molecules]):

        assert type(chemspace) == list, "Chemspace must be a list of Molecules objects."
        assert len(chemspace) > 0, "Chemspace must contain at least one molecule."
        assert all(isinstance(n, Molecules) for n in chemspace), "Chemspace must be a list of Molecules objects."


        self.hash_space = dict()
        self.index_space = list()
        self.NumOfProperties = list()

        # Create a dictionary of Molecules objects, with the molecule name as the key.
        # Save the length of the molecules list as the value.
        for molecules in chemspace:
            self.hash_space[molecules.setName] = molecules
            self.index_space.append(molecules.properties.shape[0])
            self.NumOfProperties.append(molecules.properties.shape[1])

        assert len(set(self.NumOfProperties)) == 1, "All the sets of molecules must have the same number of properties."

        # Dimensionality reduction configuration
        self.nComponents = None


        self.properties = np.vstack([x.properties for x in self.hash_space.values()])
        self.num_properties = self.properties.shape[1]
        self.num_molecules = self.properties.shape[0]
        
        # Store the PC coordinates in the Molecules objects
        self.coordinates = None
        # Store the model 
        self.model = None
        self.dataframe = None

        # create a nparray of all the setNames in the chemspace
        self.set_names = np.array([x.setName for x in self.hash_space.values() for i in range(x.num_molecules)])


    def _store_pc_coordinates(self):
        """
        For each set of <Molecules> in the chemspace, store the PCA coordinates in the <Molecule> object 
        by indexing the self.coordinates with the molecule index (self.index_space).
        """

        current_index = 0
        for molecules in self.hash_space.values():
            length = molecules.num_molecules
            # print(molecules)
            # print(f"Current index: {current_index}, length: {length}")
            molecules.coordinates = self.coordinates[np.r_[current_index:length+current_index, :]]
            current_index += length


    def tSNE(self, model='t-SNE', scaler=None, nComponents=2, perplexity=30,  
                early_exaggeration=12.0, learning_rate='warn', n_iter=1000, 
                n_iter_without_progress=300, min_grad_norm=1e-07, 
                metric='euclidean', metric_params=None, init='warn', 
                verbose=0, random_state=None, method='barnes_hut', angle=0.5, 
                n_jobs=None):
        """
        Parameters
        ==========
        model: str
            The model to be used for dimensionality reduction.
        scaler: sklearn.preprocessing.StandardScaler
            The scaler to be used for dimensionality reduction.
        Others:
            seehttps://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html for more information.
        
        Returns
        =======
        self.coordinates: numpy.ndarray
            The TSNE coordinates of the molecules.
        self.model: sklearn.manifold.TSNE
            The TSNE model.
        scaler: sklearn.preprocessing.StandardScaler
            The scaler used for dimensionality reduction.
        """

        self.nComponents = nComponents

        if model in ('t-SNE'):
            # Standardize data
            scaler = StandardScaler()
            descriptors_scaled = scaler.fit_transform(self.properties)

            # Apply selected model to data (PCA, TruncatedSVD, LinearDiscriminantAnalysis).
            m = TSNE(nComponents, perplexity, early_exaggeration, learning_rate, n_iter, n_iter_without_progress,
                        min_grad_norm, metric, metric_params, init, 
                        verbose, random_state, method, angle, n_jobs) # get model
            print("Applying {} model...".format(model))
            self.coordinates = m.fit_transform(descriptors_scaled)[:,:nComponents]
            print(f"Number of components: {self.coordinates.shape[1]}")

        else:
            assert type(model) in (TSNE), "Selected model must be TSNE from sklearn."
            m = model
            
            if scaler is None:
                #Standardize data
                scaler = StandardScaler()
                descriptors_scaled = scaler.fit_transform(self.properties)

            else:
                descriptors_scaled = scaler.transform(self.properties)
            # Apply selected model to data (PCA, TruncatedSVD, LinearDiscriminantAnalysis).
            self._coordinates = m.fit_transform(descriptors_scaled)

        self.model = m

        # Store the PCA coordinates in the Molecules objects
        self._store_pc_coordinates()
        return self.coordinates, self.model, scaler

    def PCA(self, model='PCA', scaler=None, nComponents=None, copy=True, whiten=False, svd_solver='auto', 
            tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=None):
        
        """
        Parameters
        ==========
        model: str
            The model to be used for dimensionality reduction.
        scaler: sklearn.preprocessing.StandardScaler
            The scaler to be used for dimensionality reduction.
        Others:
            see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html for more information.
        
        Returns
        =======
        self.coordinates: numpy.ndarray
            The PCA coordinates of the molecules.
        self.model: sklearn.decomposition.PCA
            The PCA model.
        scaler: sklearn.preprocessing.StandardScaler
            The scaler used for dimensionality reduction.
        """
        
        self.nComponents = nComponents if nComponents is not None else self.num_properties
        
        if model in ('PCA'):
            if scaler is None:
                #Standardize data
                scaler = StandardScaler()
                descriptors_scaled = scaler.fit_transform(self.properties)
            else:
                descriptors_scaled = scaler.transform(self.properties)

            # Apply selected model to data (PCA).
            self.model = PCA(n_components=self.nComponents, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, 
            iterated_power=iterated_power, n_oversamples=n_oversamples, power_iteration_normalizer=power_iteration_normalizer, 
            random_state=random_state) 

        else:
            assert type(model) in (PCA), "Enter a pretrainded PCA model."
            self.model = model
            if scaler is None:
                #Standardize data
                scaler = StandardScaler()
                descriptors_scaled = scaler.fit_transform(self.properties)
            
            else:
                descriptors_scaled = scaler.transform(self.properties)
            
        print("Applying PCA model...")
        self.coordinates = self.model.fit_transform(descriptors_scaled)[:,:nComponents]

        print("{}% variance explained with {} components".format(np.round(np.sum(self.model.explained_variance_ratio_[:self.nComponents]), 3)*100, self.nComponents ) )

        plot_explained_variance(self.model, self.nComponents)
        # Store the PCA coordinates in the Molecules objects
        self._store_pc_coordinates()

        return self.coordinates, self.model, scaler #, explained_variance

    def __str__(self):
        
        return "<ChemSpace> object with {} sets of molecules.\nChemSpace dimensions: {}".format(len(self.hash_space.keys()), self.properties.shape)

    def to_df(self):
        """
        Convert the ChemSpace object to a pandas.DataFrame.
        It uses the to_df() method of Molecule objects to create the DataFrame.
        """
        assert self.coordinates is not None, "PCA or TSNE coordinates must be computed first."

        dfs = list()
        for Ms in self.hash_space.values():
            dfs.append(Ms.to_df())
        df = pd.concat(dfs, ignore_index=True)
        self.dataframe = df
        return df    

    def to_sdf(self, filename, molCol='ROMol', propsCols=None):
        """
        Convert the ChemSpace object to an sdf file.
        """
        if self.dataframe is None:
            df = self.to_df()
            propsCols = list(df.columns)
            PandasTools.WriteSDF(df , f'{filename}.sdf', molColName = molCol, properties = propsCols)
        else:
            propsCols = list(self.dataframe.columns)
            PandasTools.WriteSDF(self.dataframe , f'{filename}.sdf', molColName = molCol, properties = propsCols)
        
        print("SDF file saved to {}.sdf\n{} Molecules converted.".format(filename, self.num_molecules))





























import os

import pickle
import uuid
import time

from typing import Tuple, Union

import pandas as pd
import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools, Descriptors

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cluster import KMeans


from scipy.spatial import distance

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def compute_distance(space_ref: np.array, space_target: np.array) -> float:
    """
    Compute min distance between all molecules in space_ref and space_comp.

    Parameters
    ----------
    space_ref : np.array
        numpy array containing dimensional coordinates of molecules in space_ref.
    space_target : np.array
        numpy array containing dimensional coordinates of molecules in space_target.

    Returns
    -------
    float
        minimum distance between space_ref and space_target
    """

    distance_matrix = distance.cdist(space_ref, space_target, 'euclidean')

    return distance_matrix.min(axis=1)


def plot_distances(df: pd.DataFrame, ref_region: str=None, filename: str=None):
    """
    Plot the distances between the molecules in the chemspace
    and the "target" region of the chemspace.

    Parameters
    ----------
        df : pd.DataFrame
            pandas dataframe containing the coordinates of the molecules.
        ref_region : str
            name of the reference region.
        filename : str
            name of the file to save the plot.        
    """
            
    region = df[df.setName == ref_region]
    allsets = df[df.setName != ref_region]
    
    fig, ax = plt.subplots(1,1, figsize=(10, 6))
    plt.grid()

    ax1 = ax.scatter(data=region, x='PC1', y='PC2', marker='o', s=125, color="whitesmoke", alpha=1)
    ax2 = ax.scatter(data=allsets, x='PC1', y='PC2', marker='+', c=allsets['EuclDist'], cmap='plasma')
    ax3 = plt.scatter(data=allsets[allsets.setName == "IPTACOPAN"],
                    x='PC1',
                    y='PC2',
                    marker='X',
                    s=223)

    cbar = fig.colorbar(ax2, ax=ax)
    cbar.set_label('Euclidean Distance')

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    
    # LEGEND
    patch1 = mpatches.Patch(color='whitesmoke', label='Target Chemical Space')

    ipt = mlines.Line2D([], [], color='indigo', marker='X', linestyle='None',
                            markersize=10, label='Iptacopan', markeredgewidth=4)

    plt.legend(handles=[patch1, ipt], frameon=False, loc='upper left')
    plt.tight_layout()
    if filename is not None:
         plt.savefig(f"{filename}_distances.png", dpi=300)
    plt.show()

    # Histogram of distances by set name
    ax = allsets.pivot(columns='setName', values='EuclDist').plot(kind='hist', 
                              bins = 100, 
                              figsize=(12,8),
                              alpha = 0.6, grid=True)
    vline = ax.vlines(x = 2.525, ymin = 0, ymax = 80,
            colors = 'black',
            label = 'Iptacopan')
    ax.set_xlabel("Euclidean Distances")
    leg1 = ax.legend(['CFB', 'Enamine Real', 'Enamine Real - In Space', 'Iptacopan scaffold'], loc='upper right')
    leg2 = ax.legend(handles=[vline], loc='lower right')
    ax.add_artist(leg1)
    if filename is not None:
        plt.savefig(f"{filename}_hist.png", dpi=300)
    plt.show()


def plot_explained_variance(pca, dims):

    plt.bar(range(1,dims+1), pca.explained_variance_ratio_,
            alpha=0.5,
            align='center')
    plt.step(range(1,dims+1), np.cumsum(pca.explained_variance_ratio_),
            where='mid',
            color='red')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal Components')
    plt.show()


def coords_atoms(mol: Chem.rdchem.Mol) -> Tuple[list, np.array]:
    """
    Compute the coordinates of the atoms in a molecule.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        rdkit molecule.

    Returns
    -------
    list
        list of atom coordinates.
    np.array
        numpy array containing the coordinates of the atoms.
    """


    atoms_ = []
    coords_ = []
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        positions = conf.GetAtomPosition(i) # Get the coordinates of the atom
        atoms_.append(atom) # append atom object
        coords_.append(np.array((positions.x, positions.y, positions.z))) # append coordinates

    coords_ = np.vstack([x for x in coords_]) # stack coordinates
    
    return atoms_, coords_

def find_nearest_coord(frag_coords: np.array, ref_coords: np.array) -> Tuple[int, np.array, int, np.array]:
    """
    Find the nearest coordinate in the reference region to a coordinate in the fragment region.

    Parameters
    ----------
    frag_coords : np.array
        numpy array containing the coordinates of the fragment region.
    ref_coords : np.array
        numpy array containing the coordinates of the reference region.
    
    Returns
    -------
    int
        index of the nearest coordinate in the reference region.
    np.array
        numpy array containing the coordinates of the nearest coordinate in the reference region.
    int
        index of the nearest coordinate in the fragment region.
    np.array
        numpy array containing the coordinates of the nearest coordinate in the fragment region.
    """


    distance_matrix = distance.cdist(ref_coords, frag_coords, 'euclidean')
    frag_dist = distance_matrix.min(axis=0)
    ref_dist = distance_matrix.min(axis=1)
    
    # Index of "ith" element of frag_coords having min distance
    index1 = np.where(frag_dist == np.amin(frag_dist))
    # Index of "ith" element of ref_coords having min distance
    index2 = np.where(ref_dist == np.amin(ref_dist))
    
    return index2[0][0], ref_coords[index2[0][0]], \
           index1[0][0], frag_coords[index1[0][0]]


def combine(ref: Chem.rdchem.Mol, frag: Chem.rdchem.Mol, i: int, j: int) -> Chem.rdchem.Mol:
    """
    Combine two molecules by adding the fragment to the reference region using the indeces of the nearest atom.

    Parameters
    ----------
    ref : Chem.rdchem.Mol
        rdkit reference molecule.
    frag : Chem.rdchem.Mol
        rdkit fragment molecule.
    
    Returns
    -------
    Chem.rdchem.Mol
        rdkit molecule containing the combined molecules.
    """
    # Create a combined molecule
    combo = Chem.CombineMols(ref, frag)
    emol = Chem.EditableMol(combo)

    # num will be the index of the last atom in the reference region
    num = ref.GetNumAtoms()

    # Add a single bond between the two atoms using the indeces of the nearest atom
    emol.AddBond(i, num+j, order=Chem.rdchem.BondType.SINGLE)

    # Convert the combined molecule to a rdkit molecule
    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return mol