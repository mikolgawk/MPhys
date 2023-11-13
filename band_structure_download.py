"""
This example prints out a band structure object of a Materials Project entry.
To run this example, you should:
* have pymatgen (www.pymatgen.org) installed
* obtain a Materials Project API key (https://www.materialsproject.org/open)
* paste that API key in the MAPI_KEY variable below, e.g. MAPI_KEY = "foobar1234"
as well as:
* update MP_ID with the Materials Project id of the compound
For citation, see https://www.materialsproject.org/citing
"""

from pymatgen.ext.matproj import MPRester
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.core import SiteCollection
import json
import os
import time
import numpy as np

# Class dealing with the conversion of different objects to json format
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    MAPI_KEY = "bbiUJxPj5bmp31SqSsuc0EAcO3ekNw6f"  # You must change this to your Materials API key! (or set MAPI_KEY env variable)
    mpr = MPRester(MAPI_KEY)  # object for connecting to MP Rest interface
    
    for id_ in range(1,100):    
        mp_id = "mp-" + str(id_)  # You must change this to the mp-id of your compound of interest
        has_bs = False
        try:
            bs = mpr.get_bandstructure_by_material_id(mp_id) # Get bandstructure by material id
            if bs is not None: has_bs = True
        except:
            pass
        if has_bs:
            try:
                data=bs.as_dict()
                print('Band gap info: {}'.format(bs.get_band_gap()))
                print(id_)
                #bs_dict = bs.as_dict()
                # Get only the information needed for the band structures images
                bs_dict2={
                    "labels_dict" : data["labels_dict"],
                    "kpoints" : data["kpoints"],
                    "efermi" : data["efermi"],
                    "bands" : data["bands"]
                }


                if data['is_spin_polarized']==False: 
                    filename = f'{mp_id}'

                    # Specify the directory where you want to save the JSON file
                    directory1 = "/home/mikolaj/MPhys_project/practise2/"
                    json_filename1 = os.path.join(directory1, f"{filename}.json")
                    #json.dump(data, json_filename1, cls=NpEncoder)
                            #BSPlotter(bs).save_plot(filename)
                    with open(json_filename1, "w") as json_file:
                        json.dump(bs_dict2, json_file, cls=NpEncoder, indent=2)
                else:
                    print("Band structure not downloaded")

            except:
                pass  
        