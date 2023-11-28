"""Obtain the affiliations of the first authors of K2 papers, and match
to Carnegie Classifications where possible.

Carnegie Classifications are obtained from:
https://carnegieclassifications.acenet.edu/downloads.php
"""
import os, sys

import nltk
import time
from tqdm import tqdm
import kpub
import matplotlib.pyplot as plt
from astropy.table import Table,join
import numpy as np
import astropy.io.ascii as at

# For carnegie classifications, have to assume some equivalences
cequiv0 = at.read("equivalencies.csv")
# cequiv = Table(cequiv0, masked=False, copy=True)
cequiv = cequiv0.filled(fill_value=-9999)
# print(cequiv["MergeName"])
# sys.exit()

cequiv_names = cequiv["Affiliation"][cequiv["MergeName"]!="-9999"]

# Step 1: obtain the first author affiliations from kpub
def get_locations(mission=None,n_show=20):
    if mission is not None:
        print("\n\n",mission)
        output_filename = f"affiliations_{mission}.csv"
    else:
        print("\n\nAll Results")
        output_filename = f"affiliations_all.csv"

    locations = []
    db = kpub.PublicationDB()
    all_publications = db.get_all(mission=mission)
    # Find affiliations for authors of papers from the chosen mission
    for publication in all_publications:
        affiliations = publication['aff']
        # ****Right now this is still VERY kludgy. *****
        # Some institutions are double-counted, many things that aren't institutions
        # like streets are included

        # TODO: use the geolocation from another script to separate international institutions

        # Use the first three authors
        pub_affil = []
        for aff in affiliations[:3]:
            # I want only the name of the institutions
            aff0 = aff.split(";")[0].split(",")
            for aff_suffix in aff0:
                if (("Depart" in aff_suffix) or ("Str." in aff_suffix) or
                    ("Street" in aff_suffix) or ("Rue" in aff_suffix) or
                    ("Ave." in aff_suffix) or ("Ave " in aff_suffix) or
                    ("College St" in aff_suffix) or ("Drive" in aff_suffix) or
                    ("Avenue" in aff_suffix) or ("Centre-ville" in aff_suffix)
                    or ("Camino" in aff_suffix)):
                    # I don't want department names or addresses
                    continue
                elif (("Univers" in aff_suffix) or ("College" in aff_suffix) or
                      ("Center" in aff_suffix) or ("Instit" in aff_suffix) or
                      ("Centre" in aff_suffix) or ("Observ" in aff_suffix)):

                    base_location = aff_suffix.replace("The","").replace("\"","").strip()

                if base_location in cequiv_names:
                    loc = np.where(cequiv["Affiliation"]==base_location)[0]
                    if len(loc)>1:
                        loc = loc[0]
                    if cequiv["MergeName"][loc][0]!="-9999":
                        location = cequiv["MergeName"][loc][0]
                    else:
                        location = base_location
                else:
                    location = base_location

                if ";" in location:
                    loc1, loc2 = location.split(";")
                    pub_affil.append(loc1)
                    pub_affil.append(loc2)
                else:
                    pub_affil.append(location)

            pub_unique = np.unique(pub_affil)

        if len(pub_unique)>0:
            locations.append(pub_unique)
                    # break
    locations = np.concatenate(locations)
    unique_locations = nltk.FreqDist(locations)
    print("Found {} unique locations".format(len(unique_locations)))

    # Write the affiliations out to a file for later reference
    f = open(output_filename,"w")
    if mission is not None:
        f.write(f"# Affiliations for {mission}, first three authors of each paper\n")
    f.write("affiliation,count\n")
    for item in unique_locations.items():
        f.write(str(item[0]))
        f.write(",")
        f.write(str(item[1]))
        f.write("\n")
    f.close()

    # Return arrays of the affiliation names and their occurrences
    institutions = np.array([item[0] for item in unique_locations.items()])
    counts = np.array([item[1] for item in unique_locations.items()])

    return institutions, counts

def carnegie_match(institutions):
    """Given a list of institution names, find the Carnegie classification"""

    car = at.read("CCIHE2021-PublicData.csv")

    cclass = np.ones(len(institutions))*-2

    for i,inst in enumerate(institutions):
        found = False
        fname = inst
        if inst in car["name"]:
            found = True

        if (found==False) and (inst in cequiv["Affiliation"]):
            loc = np.where(cequiv["Affiliation"]==inst)[0]
            if len(loc)>=1:
                if len(loc)>1:
                    loc = loc[0]
                fname = cequiv["MergeName"][loc][0]
                found = True


        if found==False:
            test_names = [f"The {inst}",inst.replace(" at ","-"),
                          inst.replace("-"," at "),inst.replace("-"," "),
                          inst.replace(" at "," "),
                          inst.replace("'",""),inst.replace("`",""),
                          inst.replace("State University of New York","SUNY"),
                          inst.replace(" &amp","")]
            for name in test_names:
                if name in car["name"]:
                    found = True
                    fname = name


        # if found==False:
        #     flist = []
        #     for cname in car["name"]:
        #         if inst in cname:
        #             flist.append(cname)
        #     if len(flist)==1:
        #         found = True
        #         fname = flist[0]
        #     elif (len(flist)>0) and (len(flist)<25):
        #         print("\n",inst)
        #         print(flist)

        loc = np.where(car["name"]==fname)[0]
        loc2 = np.where((cequiv["Affiliation"]==fname) & (cequiv["CarnegieClass"]>-9998))[0]
        loc3 = np.where((cequiv["MergeName"]==fname) & (cequiv["CarnegieClass"]>-9998))[0]
        # print(inst,loc,loc2,loc3)
        if len(loc)==1:
            cclass[i] = car["basic2021"][loc]
        elif len(loc2)>0:
            if len(loc2)>1:
                loc2 = loc2[0]
            cclass[i] = cequiv["CarnegieClass"][loc2]
        elif len(loc3)>0:
            if len(loc3)>1:
                loc3 = loc3[0]
            cclass[i] = cequiv["CarnegieClass"][loc3]
        elif ("Coll" in inst) or ("Univ" in inst):
            # print("Not found:",inst)
            pass
        else:
            # Assume in this case, it's a street name or department name
            continue

    return cclass

def make_table():
    """ Make a table showing the paper counts by institution for Kepler & K2 """
    k2_inst, k2_ct = get_locations(mission="k2")
    kep_inst, kep_ct = get_locations(mission="kepler")

    k2_tab = Table({"Affiliation":k2_inst,"K2_count":k2_ct})
    kep_tab = Table({"Affiliation":kep_inst,"Kepler_count":kep_ct})

    tab = join(k2_tab,kep_tab,join_type="outer")
    tab.sort(["K2_count","Kepler_count"])
    tab.reverse()

    cclass = carnegie_match(tab["Affiliation"])
    tab["CarnegieClass"] = cclass

    tab.write("affiliations_kepler_k2.csv",delimiter=",",overwrite=True)
    return tab

def carnegie_counts(tab):
    """Given a table from make_table(), count publications by carnegie class"""

    # Anything with a class of -2 is not included in Carnegie Classification
    # either according to the original table, or because the function above
    # couldn't find a match. The real classifications start at 1.
    noclass = np.where(tab["CarnegieClass"]<0)[0]
    # tab = tab[(tab["CarnegieClass"]>0) & np.isfinite(tab["CarnegieClass"])]
    tab = tab[np.isfinite(tab["CarnegieClass"])]

    # Read in the table of values
    val = at.read("CCIHE2021_BASIC2021Classifications.csv")
    # print(val.dtype)

    # classes = np.sort(np.unique(tab["CarnegieClass"]))
    classes = val["Value"]
    k2_counts = np.zeros(len(classes),"int")
    kep_counts = np.zeros(len(classes),"int")

    k2_col = tab["K2_count"].filled(0)
    kep_col = tab["Kepler_count"].filled(0)

    for i,cls in enumerate(classes):
        loc = np.where((tab["CarnegieClass"]==cls) & np.isfinite(tab["CarnegieClass"]))[0]
        if len(loc)>0:
            k2_counts[i] = np.nansum(k2_col[loc])
            kep_counts[i] = np.nansum(kep_col[loc])
            print("\n",str(val["Label"][val["Value"]==cls]))
            print("Kepler:",kep_counts[i],"K2:",k2_counts[i])

    val["K2_count"] = k2_counts
    val["Kepler_count"] = kep_counts

    val.write("affiliations_carnegie.csv",delimiter=",",overwrite=True)


if __name__=="__main__":

    # fname = "affiliations_kepler_k2_new.csv"
    # if os.path.exists(fname):
    #     tab = at.read(fname)
    # else:
    tab = make_table()

    if tab.masked == True:
        tab = Table(tab, masked=False, fill_value=0)

    carnegie_counts(tab)

    # cclass = carnegie_match(["Lafayette College","Columbia University",
    #                         "Columbia University in the City of New York"])
