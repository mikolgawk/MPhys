# """
# Use this code to obtain Inorganic Crystal Structure Database IDs for Materials Project entries
# """

from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.core import SiteCollection
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester
from pymatgen.core.lattice import Lattice
# from pymatgen.ext.matproj import MPRester
import json
import os
import time
import numpy as np
import pickle

materials = []

# for i in range(0,len(mat)):
#     materials.append(mat[i].material_id)

# Open the file in read mode
with open("compound_flat_ids_099_allranges.txt", "r") as file:
    # Read the lines from the file
    lines = file.readlines()

# Remove newline characters and create a list
materials = [line.strip() for line in lines]

with MPRester('58kPxqejOOHdX2wVbsu0pW8NyY4z7NDx') as mpr:
    doc = [mpr.provenance.get_data_by_id(mp_id, fields=['material_id', 'database_IDs']) for mp_id in materials]

with open('file_flats_icsd_compound_099_allranges.txt','w') as file:
    for i in range(len(materials)):
        try:
            file.write(str(doc[i].database_IDs['icsd']))
            file.write('\n')
        except:
            pass




