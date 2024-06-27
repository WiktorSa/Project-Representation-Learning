#!/usr/bin/python

# Based on https://www.kaggle.com/code/mashruravi/pytorch-vs-cancer

import SimpleITK as sitk
import glob
import numpy as np
import pandas as pd
import os
import argparse

def get_class(diameter):
    return 0 if diameter < 3 else 1

def load_images_and_labels(scan_filepath, patient_data):
    # We don't want to have all the scans while training the model.
    DIMS_IRC = (10, 32, 32)
    
    mhd_file = sitk.ReadImage(scan_filepath)
    ct_scan = np.array(sitk.GetArrayFromImage(mhd_file), dtype=np.float32)

    # Clip data to avoid extreme values
    ct_scan.clip(-1000, 1000, ct_scan)
    
    origin_xyz = np.array(mhd_file.GetOrigin())
    voxel_size_xyz = np.array(mhd_file.GetSpacing())
    direction_matrix = np.array(mhd_file.GetDirection()).reshape(3, 3)

    # Some patients have multiple nodules
    images = []
    labels = []
    for _, row in patient_data.iterrows():
        coordX = row['coordX']
        coordY = row['coordY']
        coordZ = row['coordZ']
        class_id = row['class']

        center_xyz = np.array([coordX, coordY, coordZ])

        # We want to get the most important elements from our scans 
        cri = ((center_xyz - origin_xyz) @ np.linalg.inv(direction_matrix)) / voxel_size_xyz
        cri = np.round(cri)

        irc = (int(cri[2]), int(cri[1]), int(cri[0]))

        slice_list = []
        for axis, center_val in enumerate(irc):
            start_index = int(round(center_val - DIMS_IRC[axis]/2))
            end_index = int(start_index + DIMS_IRC[axis])
        
            if start_index < 0:
                start_index = 0
                end_index = int(DIMS_IRC[axis])
            
            if end_index > ct_scan.shape[axis]:
                end_index = ct_scan.shape[axis]
                start_index = int(ct_scan.shape[axis] - DIMS_IRC[axis])
                
            slice_list.append(slice(start_index, end_index))

        # Slice data - get the most valuable information from the most valuable photos
        ct_scan_chunk = ct_scan[tuple(slice_list)]

        images.append(ct_scan_chunk)
        labels.append(class_id)

    return images, labels

def main(annotation_file, annotations_excluded_file, image_location, subset, data_save_folder):
    df_ann = pd.read_csv(annotation_file)
    df_irrelevant_findings = pd.read_csv(annotations_excluded_file)
    
    df = pd.concat([df_ann, df_irrelevant_findings])
    df['class'] = df['diameter_mm'].apply(lambda x: get_class(x))
    
    patients_data = df[['seriesuid', 'coordX', 'coordY', 'coordZ', 'class']]

    filepaths = glob.glob(f'{image_location}/subset{subset}/*.mhd')
    filenames = [x.replace(f'{image_location}/subset{subset}/', '').replace('.mhd', '') for x in filepaths]
    
    all_images = []
    all_labels = []
    for filename, filepath in zip(filenames, filepaths):
        patient_data = patients_data[patients_data['seriesuid'] == filename].reset_index(drop=True)
        if patient_data.shape[0] != 0:
            images, labels = load_images_and_labels(filepath, patient_data)
            all_images.extend(images)
            all_labels.extend(labels)
            
    os.makedirs(f"{data_save_folder}/subset{subset}", exist_ok=True)
    
    # Shape - No scans slices x channels x no images (planes) x height x width
    all_images_numpy = np.array(all_images)
    all_images_numpy = np.expand_dims(all_images_numpy, axis=1)
    
    # Shape - No scans
    all_labels_numpy = np.array(all_labels)
    
    np.savez(f"{data_save_folder}/subset{subset}/data.npz", images=all_images_numpy, labels=all_labels_numpy)

# Usage - python process_data.py
if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotations', help="Location of .csv file with annotations", default="data/annotations.csv", type=str)
    parser.add_argument('-an', '--annotations_excluded', help="Location of .csv file with excluded annotations", default="data/annotations_excluded.csv", type=str)
    parser.add_argument('-i', '--images', help="Location of the CT images", default="data/images", type=str)
    parser.add_argument('-s', '--subset', help="Number of subset to load data from", default=0, type=int)
    parser.add_argument('--save', help="Where to save data", default="data_transform", type=str)
    args = parser.parse_args()
    main(args.annotations, args.annotations_excluded, args.images, args.subset, args.save)
