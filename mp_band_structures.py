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

if __name__ == "__main__":
    MAPI_KEY = "bbiUJxPj5bmp31SqSsuc0EAcO3ekNw6f"  # You must change this to your Materials API key! (or set MAPI_KEY env variable)
    mpr = MPRester(MAPI_KEY)  # object for connecting to MP Rest interface
    
    for id_ in range(1,5):    
        mp_id = "mp-" + str(id_)  # You must change this to the mp-id of your compound of interest
        has_bs = False
        try:
            bs = mpr.get_bandstructure_by_material_id(mp_id)
            if bs is not None: has_bs = True
        except:
            pass
        if has_bs:
            print('Band gap info: {}'.format(bs.get_band_gap()))
            print(id_)
            filename = f'band_structure_{mp_id}'
            BSPlotter(bs).save_plot(filename)
              
        
        
        #mpr = MPRester(MAPI_KEY)  # object for connecting to MP Rest interface

        #my_bs = mpr.get_bandstructure_by_material_id(MP_ID)
        #print('Band gap info: {}'.format(my_bs.get_band_gap()))
        #BSPlotter(my_bs).show()