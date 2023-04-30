import pandas as pd 
import os 
#import 

RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/manual_measurements_testing/results/"
DATA_PATH= "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers_manual/manual_2/"

OG_F = os.listdir(RES_PATH)

for f in OG_F:
    file = RES_PATH+ f
    if f.endswith(".csv") and "marilin" in f:
        core_name = f.split(".csv")[0]
        dms = pd.read_csv(file)["Length"]
        with open(f"{DATA_PATH}{core_name}.txt", "w+") as file:

               file.write("***diameter values***")
               file.write("\n")
               for val in dms:
                    file.write(f"{val}")
                    file.write("\n")
        #print(f)
    #print(f)
#pd.read_csv()