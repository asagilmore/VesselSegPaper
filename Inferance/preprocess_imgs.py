import os
import argparse
import ants
from tqdm import tqdm

try:
    import antspynet
except ImportError:
    antspynet = None

ref_img = ants.image_read('ref.nii.gz')


def extract_brain(img, modality='t1'):
    probability_mask = antspynet.brain_extraction(img, modality=modality)
    brain_mask = ants.get_mask(probability_mask, low_thresh=0.5)
    brain_extracted = img * brain_mask
    return brain_extracted


def process_img(img_pth, output_dir, file_name, skull_strip=False):
    img = ants.image_read(img_pth)

    if skull_strip:
        img = extract_brain(img)

    img_reg = ants.registration(fixed=ref_img, moving=img,
                                type_of_transform='Rigid')
    img = img_reg['warpedmovout']

    img = ants.resample_image_to_target(img, ref_img, interp_type='linear')
    ants.image_write(img, os.path.join(output_dir, file_name))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--t1_dir", type=str, required=True)
    args.add_argument("--t2_dir", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True)
    args.add_argument("--skull_strip", type=bool, default=False)
    args.add_argument("--id_delim", type=str, default='_')

    args = args.parse_args()

    if args.t1_dir is not None:
        t1_files = [f for f in os.listdir(args.t1_dir) if f.endswith('.nii.gz')]

    if args.t2_dir is not None:
        t2_files = [f for f in os.listdir(args.t2_dir) if f.endswith('.nii.gz')]

    if args.skull_strip:
        if antspynet is None:
            raise ImportError("antspynet is required for skull stripping")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test = t1_files[0]
    subid = test.split(args.id_delim)[0]
    file_name = f'{subid}_0000.nii.gz'
    if '-' in subid or '_' in subid:
        print(f"Warning: {subid} contains a '-' or '_' character. This will "
              "cause issues with nnUNet, please remove these characters from "
              "the subject ID or adjust the id_delim argument.")

    for t1_file in tqdm(t1_files):
        subid = t1_file.split(args.id_delim)[0]
        t2_files = [f for f in t2_files if subid in f]
        if len(t2_files) != 1:
            print(f"Expected 1 T2 file for {subid}, found "
                  f"{len(t2_files)}, skipping")
            continue

        t2_file = t2_files[0]
        t1_pth = os.path.join(args.t1_dir, t1_file)
        t2_pth = os.path.join(args.t2_dir, t2_file)

        t1_file_name = f'{subid}_0000.nii.gz'
        t2_file_name = f'{subid}_0001.nii.gz'

        process_img(t1_pth, args.output_dir, t1_file_name, args.skull_strip)
        process_img(t2_pth, args.output_dir, t2_file_name, args.skull_strip)
