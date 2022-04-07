#! /bin/bash

#  to create the true segementations
python segment.py process_scene --overwrite --nofrom_s3 --data_folder_name=training --limit=2 --name=demo --nocfar

#  to create the true CFAR based segmentation
python segment.py process_scene --overwrite --nofrom_s3 --data_folder_name=training --limit=2 --name=demo --cfar
