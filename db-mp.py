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
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json


if __name__ == "__main__":
    MAPI_KEY = "bbiUJxPj5bmp31SqSsuc0EAcO3ekNw6f"  # You must change this to your Materials API key! (or set MAPI_KEY env variable)
    mpr = MPRester(MAPI_KEY)  # object for connecting to MP Rest interface
    json_filename = f"bs.json"
    with open(json_filename, "w") as json_file:
        for id_ in range(1,5):    
            mp_id = "mp-" + str(id_)  # You must change this to the mp-id of your compound of interest
            has_bs = False
            try:
                bs = mpr.get_bandstructure_by_material_id(mp_id)
                s = mpr.get_structure_by_material_id(mp_id)
                ## Band gap
                b_gap = bs.get_band_gap()
                ## Space group number
                sg_n = s.get_space_group_info()[1]
                ##
                docs = mpr.summary.search(material_ids=[mp_id])
                example_doc = docs[0]
                mpid = example_doc.material_id
                formula = example_doc.formula_pretty

                ### Space group info
                sg_analyzer = SpacegroupAnalyzer(s)
                crystal_s = sg_analyzer.get_crystal_system()
                group_s = sg_analyzer.get_space_group_symbol()
                number_s = sg_analyzer.get_space_group_number()
                point_g = sg_analyzer.get_point_group_symbol()

                ### Elements
                
                ### Discovery_process (is it even necessary?)

                ### Formula_anonymous (is it even necessary?)

                D={
                    "material_id" : mpid,
                    "formula_pretty" : formula,
                    "sg_number" : sg_n,
                    "bandgap" : b_gap["energy"],
                    "crystal_system" : crystal_s,
                    "point_group" : point_g
                }
            
                
                if bs is not None: has_bs = True
            except:
                pass
            if has_bs:
                print('Band gap info: {}'.format(bs.get_band_gap()))
                print(id_)
                filename = f'band_structure_{mp_id}'
                #json_filename = f"{mp_id}_bs.json"
                json.dump(D, json_file) 
                json_file.write("\n")

        
        
        #mpr = MPRester(MAPI_KEY)  # object for connecting to MP Rest interface

        #my_bs = mpr.get_bandstructure_by_material_id(MP_ID)
        #print('Band gap info: {}'.format(my_bs.get_band_gap()))
        #BSPlotter(my_bs).show()