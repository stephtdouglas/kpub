"""Obtain the affiliations of the first authors of K2 papers, and match
to Carnegie Classifications where possible.

Carnegie Classifications are obtained from:
https://carnegieclassifications.acenet.edu/downloads.php
"""
import os

import nltk
import time
from tqdm import tqdm
import kpub
import matplotlib.pyplot as plt
from astropy.table import Table,join
import numpy as np
import astropy.io.ascii as at

# For carnegie classifications, have to assume some equivalences
cequiv = {"University of Texas":"The University of Texas at Austin",
          "University of Texas-San Antonio":"The University of Texas at San Antonio",
          "University of Texas at Austin 2515 Speedway":"The University of Texas at Austin",
          "University of Virginia":"University of Virginia-Main Campus",
          "Saint Mary's University":"Saint Mary's University of Minnesota",
          "University of Maryland College Park":"University of Maryland-College Park",
          "University of Maryland":"University of Maryland-College Park",
          "University of Hawai`i":"University of Hawaii at Manoa",
          "University of Hawaíi":"University of Hawaii at Manoa",
          "University of Hawaii":"University of Hawaii at Manoa",
          "University of Hawai'i":"University of Hawaii at Manoa",
          "University of Hawaii at Mānoa":"University of Hawaii at Manoa",
          "University of Hawaií at Manoa":"University of Hawaii at Manoa",
          "University of Idaho Moscow":"University of Idaho",
          "University of Illinois at Urbana-Champaign":"University of Illinois Urbana-Champaign",
          "University of California Santa Cruz":"University of California-Santa Cruz",
          "University of Washington Box 951580":"University of Washington-Seattle Campus",
          "University of Washington":"University of Washington-Seattle Campus",
          "University of Nevada":"University of Nevada-Las Vegas",
          "Bowling Green State University":"Bowling Green State University-Main Campus",
          "Purdue University":"Purdue University-Main Campus",
          "University of Hawai’i":"University of Hawaii at Manoa",
          "University of Hawai'i-Manoa":"University of Hawaii at Manoa",
          "Washington University":"Washington University in St Louis", # Probably the biggest guess here
          "Georgia College":"Georgia College & State University",
          "Georgia College &amp":"Georgia College & State University",
          "California State Polytechnic University":"California State Polytechnic University-Pomona",
          "University of Nebraska":"University of Nebraska-Lincoln",
          "University of Pittsburgh":"University of Pittsburgh-Pittsburgh Campus",
          "University of Cincinnati":"University of Cincinnati-Main Campus",
          "NASA Goddard and University of Maryland":"University of Maryland-College Park",
          "New Mexico State University—DACC":"New Mexico State University-Dona Ana",
          "New Mexico State University-DACC":"New Mexico State University-Dona Ana",
          "the Pennsylvania State University":"The Pennsylvania State University",
          "University of California":"University of California-Berkeley",
          "Ohio State University":"Ohio State University-Main Campus",
          "Columbia University":"Columbia University in the City of New York",
          "Missouri State University":"Missouri State University-Springfield",
          "Missouri StateUniversity":"Missouri State University-Springfield",
          "University of North Carolina":"University of North Carolina at Chapel Hill",
          "University of Michigan":"University of Michigan-Ann Arbor",
          "New Mexico State University":"New Mexico State University-Main Campus",
          "Louisiana State University":"Louisiana State University and Agricultural & Mechanical College",
          "University of Minnesota":"University of Minnesota-Twin Cities",
          "Arizona State University":"Arizona State University Campus Immersion",
          "University of Oklahoma":"University of Oklahoma-Norman Campus",
          "University of Colorado":"University of Colorado Boulder",
          "University of California San Diego":"University of California-San Diego",
          "Indiana University":"'Indiana University-Bloomington",
          "University of Wisconsin - Madison":"University of Wisconsin-Madison",
          "University of Wisconsin─Madison":"University of Wisconsin-Madison",
          "University of Wisconsin—Madison":"University of Wisconsin-Madison",
          # "State University of New Jersey":"",
          "Embry-Riddle Aeronautical University":"Embry-Riddle Aeronautical University-Prescott",
          "University of California Berkeley":"University of California-Berkeley",
          "Hobart and William Smith Colleges":"Hobart William Smith Colleges",
          "Cornell Center for Astrophysics and Planetary Science":"Cornell University"
          # "":"",
          # "":"",
          # "":"",
          # "":"",
          # "":""
          }

cequiv_names = cequiv.keys()

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

        # Use the first three authors
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

                    locations.append(aff_suffix.replace("The","").replace("\"","").strip())
                    # break
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

        if (found==False) and (name in cequiv_names):
            name = cequiv[inst]
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
        if len(loc)==1:
            cclass[i] = car["basic2021"][loc]
        elif ("Coll" in inst) or ("Univ" in inst):
            print("Not found:",inst)
        else:
            # Assume in this case, it's a street name or department name
            continue

    # Now assign values for a handful of known US-based but non-university locations
    # -3 will be the flag here
    to_add = ["Harvard-Smithsonian Center for Astrophysics",
              "NASA Ames Research Center","Harvard‚îÄSmithsonian Center for Astrophysics",
              "Spitzer Science Center (SSC)","NASA Goddard Space Flight Center",
              "Infrared Processing and Analysis Center (IPAC)","NASA Exoplanet Science Institute",
              "Center for Computational Astrophysics","Caltech/IPAC-NASA Exoplanet Science Institute",
              "Cerro Tololo Inter-American Observatory","SETI Institute",
              "Spitzer Science Center","Space Telescope Science Institute",
              "Southwest Research Institute","Planetary Science Institute",
              "Observatories of the Carnegie Institution for Science",
              "Infrared processing and Analysis Center (IPAC)",
              "American Association of Variable Star Observers (AAVSO)",
              "Center for Astrophysics ‚à£ Harvard &amp",
              "Gemini Observatory","Las Cumbres Observatory",
              "NASA Astrobiology Institute's Virtual Planetary Laboratory",
              "Harvard-Smithonian Center for Astrophysics",
              "National Solar Observatory","Carnegie Institution for Science",
              "Caltech/IPAC-NASA Exoplanet Science Institute Pasadena",
              "NASA/Goddard Space Flight Center",
              "NASA Goddard Space Flight Center",
              "SETI Institute/NASA Ames Research Center",
              "National Optical Astronomy Observatory",
              "Institute for Advanced Study",
              "UCO/Lick Observatory","NASA-Ames Research Center",
              "Fermilab Center for Particle Astrophysics",
              "National Center for Atmospheric Research",
              "US Naval Observatory","Harvard Smithsonian Center for Astrophysics",
              "Ames Research Center","Carnegie Observatories",
              "NASA's Goddard Space Flight Center","NASA Exoplanet Science Institute/Caltech",
              "NASA Goddard Space Flight Center (GSFC)",
              "Las Cumbres Observatory Global Telescope",
              "NASA Goddard Institute for Space Studies",
              "NASA Exoplanet Science Institute/Caltech Pasadena",
              "NASA Exoplanet Science Institute and Infrared Processing and Analysis Center",
              "NASA Astrobiology Institute‚ÄîVirtual Planetary Laboratory Lead Team",
              "NASA Herschel Science Center",
              "NASA/Ames Research Center","National Radio Astronomy Observatory",
              "Giant Magellan Telescope/Carnegie Observatories"]
    for name in to_add:
        cclass[institutions==name] = -3

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

    fname = "affiliations_kepler_k2.csv"
    if os.path.exists(fname):
        tab = at.read(fname)
    else:
        tab = make_table()

    if tab.masked == True:
        tab = Table(tab, masked=False, fill_value=0)

    carnegie_counts(tab)

    # cclass = carnegie_match(["Lafayette College","Columbia University",
    #                         "Columbia University in the City of New York"])
