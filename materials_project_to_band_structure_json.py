#from mp_api.client import MPRester
from pymatgen.ext.matproj import MPRester
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.bandstructure import BandStructure
import json
import os

if __name__ == "__main__":
    MAPI_KEY = "58kPxqejOOHdX2wVbsu0pW8NyY4z7NDx"  # You must change this to your Materials API key! (or set MAPI_KEY env variable)
    mpr = MPRester(MAPI_KEY)  # object for connecting to MP Rest interface

    for id_ in range(1,10):
        mp_id = "mp-" + str(id_)
        has_bs = False
        try:
            bs = mpr.get_bandstructure_by_material_id(mp_id)
            if bs is not None: has_bs = True
        except:
            pass
        if has_bs:
            print('Band gap info: {}'.format(bs.get_band_gap()))
            print("Item id: " + str(id_))
            filename = f'band_structure_{mp_id}'

            # Specify the directory where you want to save the JSON file
            directory = "C:/Users/Czaja/Desktop/MPhys thesis/Materials_project_picture_prep/venv/band_structure_jsons"
            json_filename = os.path.join(directory, f"{filename}.json")
            bs_dict = bs.as_dict()
            with open(json_filename, "w") as json_file:
                json.dump(bs_dict, json_file, indent=2)

            # Save plot
            #BSPlotter(bs).save_plot(filename)




    """
    #Showing the bandgap without the labels
    MP_ID = "mp-19017"  # You must change this to the mp-id of your compound of interest
    my_bs = mpr.get_bandstructure_by_material_id(MP_ID)
    print('Band gap info: {}'.format(my_bs.get_band_gap()))
    BSPlotter(my_bs).show()
    BSPlotter(my_bs).save_plot("Test picture")
    bs.asdict()
    """