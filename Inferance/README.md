### Vessel Segmentation

## Inferance Instructions

# To run inferance please first run the preprocess_imgs.py script on your data

Example: $python preprocess_imgs.py --t1_dir ./t1w --t2_dir ./t2w --output_dir ./preprocessed --id_delim "_"

This script will regester all images to a reference image, and resample them to a shape of (512,512,160)

optionally add --skull_strip if your data is not skull stripped already. This requires antspynet to be installed

# Setting up nnUNet

Install nnUNet according to their instruction found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).

Then download the model zip of choice from our repository found [here](10.6084/m9.figshare.27040633), and installing running $nnUNetv2_install_pretrained_model_from_zip model_file.zip

finally to run inferance on a preprocessed folder run on of the following three commands based on the model you downloaded:

t1 + t2 model:

$ nnUNetv2_predict -d Dataset096_IXI-costa-t2-even-split -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlans

t1 only model:

$ nnUNetv2_predict -d Dataset095_IXI-costa-even-split -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlans

t1 + t2 trained with centerline dice:

$ nnUNetv2_predict -d Dataset096_IXI-costa-t2-even-split -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainerCLDLoss -c 3d_fullres -p nnUNetResEncUNetMPlans



## Feature extraction

Feature extraction can be easily performed using the csv_from_predictions.py script. You may need to use cython to build the feature_extraction model if the prebuilt binaries do not work.
To do this navigate to the feature_extraction folder and run "$python setup.py build_ext --inplace". Once installed you can simply run "$python csv_from_predictions.py --input_dir /path/to/predictions --output_path features.csv"

