from timeit import default_timer as timer
from datetime import timedelta
import pandas as pd
import numpy as np

import csv
import pickle
from Molecule import Molecule
from rhea.db import zinc20
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw 
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
IPythonConsole.ipython_useSVG = True
IPythonConsole.molSize = (400, 400)

total_db = 0



if __name__ == "__main__":

    start = timer()

    print(f"Start Time: {timedelta(seconds=start)}\n\n")


    with zinc20.Zinc20() as db:

        
        db.cursor.execute("SELECT id, rdkit.mol_send(mol), mol as SMILE FROM instock.molecules")
        result = db.cursor.fetchall()

        assert len(result) > 0, "No molecules in the database"
        
        with open("/home/adria/zinc20_descriptors.csv", "w+") as f:
            # iterate over the results and use Molecule to compute the descriptors and save the results in a CSV file
            for res in result:
                write = csv.writer(f)
                #increment the counter of the total fragments/molecules of the DB
                total_db += 1

                
                id, mol, smile = res
                mol = Chem.Mol(mol.tobytes())

                # Compute descriptors for the molecule

                molecule = Molecule(mol)

                # save the Molecule descriptors in a CSV file
                row = np.concatenate((np.array([total_db,id,smile]),molecule.props))
                print(row, end='\r')
                write.writerow(row)

                
            
        f.close()   
    
    end = timer()
    print(f"\nElapsed time: {timedelta(seconds=end-start)}\n")
    print(f"Total size of the DB: {total_db}")

