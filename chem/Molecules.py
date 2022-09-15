
from utils import *


class Molecules:
    def __init__(self, molecules, setname=None):
        self.setName = str(uuid.uuid1()) if setname is None else setname
        self.molecules = molecules
        self.num_molecules = len(molecules)
        self.properties = None
        
        if type(molecules) == Chem.rdmolfiles.SDMolSupplier or all(isinstance(n, Chem.rdchem.Mol) for n in molecules):
            self.properties = self._calculate_properties2()
        
        else:
            self.molecules = [f"Molecule_{i+1}" for i in range(self.num_molecules)]
            self.properties = molecules
        
        # It will store the results of the Dimensionality Reduction
        self.coordinates = None

    
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
            logp = Descriptors.MolLogP(mol)
            mwt = Descriptors.MolWt(mol)
            nhbd = Descriptors.NumHDonors(mol)
            nar = Chem.Lipinski.NumAromaticRings(mol)
            pfi = logp + nar
            nhba = Descriptors.NumHAcceptors(mol)
            nradele = Descriptors.NumRadicalElectrons(mol)
            nvalele = Descriptors.NumValenceElectrons(mol)
            hamwt = Descriptors.HeavyAtomMolWt(mol)
            mabsparch = Descriptors.MaxAbsPartialCharge(mol)
            minabsparch = Descriptors.MinAbsPartialCharge(mol)
            fraCSP3 = Chem.Lipinski.FractionCSP3(mol)
            nhohc = Chem.Lipinski.NHOHCount(mol)
            noc = Chem.Lipinski.NOCount(mol)
            nalcarbo = Chem.Lipinski.NumAliphaticCarbocycles(mol)
            nalring = Chem.Lipinski.NumAliphaticHeterocycles(mol)
            nalarom = Chem.Lipinski.NumAliphaticRings(mol)
            narcarbo = Chem.Lipinski.NumAromaticCarbocycles(mol)
            narcring = Chem.Lipinski.NumAromaticHeterocycles(mol)
            nhetero = Chem.Lipinski.NumHeteroatoms(mol)
            nsatcarbo = Chem.Lipinski.NumSaturatedCarbocycles(mol)
            nsathetero = Chem.Lipinski.NumSaturatedHeterocycles(mol)
            nsatring = Chem.Lipinski.NumSaturatedRings(mol)
            nring = Chem.Lipinski.RingCount(mol)

            ps = np.asarray([logp, mwt, nhbd, nar, 
                            pfi, nhba, nradele, nvalele, hamwt, 
                            mabsparch, minabsparch, fraCSP3, nhohc,
                            noc, nalcarbo, nalring, nalarom, narcarbo,
                            narcring, nhetero, nsatcarbo, nsathetero,
                            nsatring, nring])

            mol.SetProp("MWt", str(mwt))
            mol.SetProp("logP", str(logp))
            mol.SetProp("nHBD", str(nhbd))
            mol.SetProp("NumAromaticRings", str(nar))
            mol.SetProp("PFI", str(pfi))
            mol.SetProp("nHBA", str(nhba))
            mol.SetProp("NumRadicalElectrons", str(nradele))
            mol.SetProp("NumValenceElectrons", str(nvalele))
            mol.SetProp("HeavyAtomMolWt", str(hamwt))
            mol.SetProp("MaxAbsPartialCharge", str(mabsparch))
            mol.SetProp("MinAbsPartialCharge", str(minabsparch))
            mol.SetProp("FractionCSP3", str(fraCSP3))
            mol.SetProp("NHOHCount", str(nhohc))
            mol.SetProp("NOCount", str(noc))
            mol.SetProp("NumAliphaticCarbocycles", str(nalcarbo))
            mol.SetProp("NumAliphaticHeretocycles", str(nalring))
            mol.SetProp("NumAliphaticRings", str(nalarom))
            mol.SetProp("NumAromaticCarbocycles", str(narcarbo))
            mol.SetProp("NumAromaticHeretocycles", str(narcring))
            mol.SetProp("NumHeteroatoms", str(nhetero))
            mol.SetProp("NumSaturatedCarbocycles", str(nsatcarbo))
            mol.SetProp("NumSaturatedHeterocycles", str(nsathetero))
            mol.SetProp("NumSaturatedRings", str(nsatring))
            mol.SetProp("RingCount", str(nring))


            if properties is None:
                properties = ps
                print(properties)
            else:
                properties.append(ps)

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
        df["logP"] = self.properties[:, 0]
        df["MWt"] = self.properties[:, 1]
        df["nHBD"] = self.properties[:, 2]
        df["NumAromaticRings"] = self.properties[:, 3]
        df["PFI"] = self.properties[:, 4]


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