#from utils import *
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


class Molecule:
    def __init__(self, molecule: Chem.rdchem.Mol):
        self.molecule = molecule
        self.name = molecule.GetProp("_Name")
        self.properties = self.calcProp()
        self.props = self.calcProperties()
    

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
        self.NumHDonors = Chem.Lipinski.NumHDonors(self.molecule)
        self.NumHeteroatoms = Chem.Lipinski.NumHeteroatoms(self.molecule)
        self.NumRotatableBonds = Chem.Lipinski.NumRotatableBonds(self.molecule)
        self.NumSaturatedRings = Chem.Lipinski.NumSaturatedRings(self.molecule)
        self.NumHeterocycles = Chem.Lipinski.NumHeterocycles(self.molecule)
        self.NumAromaticHeterocycles = Chem.Lipinski.NumAromaticHeterocycles(self.molecule)
        self.NumSaturatedHeterocycles = Chem.Lipinski.NumSaturatedHeterocycles(self.molecule)
        self.NumAliphaticHeterocycles = Chem.Lipinski.NumAliphaticHeterocycles(self.molecule)
        self.NumAliphaticRings = Chem.Lipinski.NumAliphaticRings(self.molecule)
        self.NHOHCount = Chem.Lipinski.NHOHCount(self.molecule)
        self.NOCount = Chem.Lipinski.NOCount(self.molecule)
        self.NumRadicalElectrons = Chem.Lipinski.NumRadicalElectrons(self.molecule)
        self.NumValenceElectrons = Chem.Lipinski.NumValenceElectrons(self.molecule)
        self.NumAliphaticCarbocycles = Chem.Lipinski.NumAliphaticCarbocycles(self.molecule)
        self.NumAromaticCarbocycles = Chem.Lipinski.NumAromaticCarbocycles(self.molecule)
        self.NumSaturatedCarbocycles = Chem.Lipinski.NumSaturatedCarbocycles(self.molecule)
        self.RingCount = Chem.Lipinski.RingCount(self.molecule)
        self.properties = np.asarray([self.logP, self.MWt, self.nHBD, self.NumAromaticRings, self.PFI,
                                     self.fracCSP3, self.NumHAcceptors, self.NumHDonors, self.NumHeteroatoms,
                                     self.NumRotatableBonds, self.NumSaturatedRings, self.NumHeterocycles,
                                     self.NumAromaticHeterocycles, self.NumSaturatedHeterocycles, self.NumAliphaticHeterocycles,
                                     self.NumAliphaticRings, self.NHOHCount, self.NOCount, self.NumRadicalElectrons,
                                     self.NumValenceElectrons, 
                                     self.NumAliphaticCarbocycles, self.NumAromaticCarbocycles, self.NumSaturatedCarbocycles,
                                     self.RingCount])
        
        return self.props        