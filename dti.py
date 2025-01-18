import os
import yaml

from preprocess import preprocess_dwi
from main import coregister_mask
from main import fit_dti

script_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.normpath(os.path.join(script_dir,'config','config.yml'))
with open(config_file_path, "r") as f: config = yaml.safe_load(f)
    
dwi_file = config["data"]["dwi_file"]
bval_file = config["data"]["bval_file"]
bvec_file = config["data"]["bvec_file"]
# fixed_file = config["data"]["fa_file"] # FA or a b0 volume
# moving_file = config["data"]["wm_mask_file"]
output_dir = config["output_dir"]

# Preprocessing
preproc_dwi, preproc_affine, mask, gtab = preprocess_dwi(
    dwi_file, bval_file, bvec_file,
    out_dir=output_dir,
    do_denoise=False,
    do_gibbs=True,
    do_motion_correction=True,
    do_masking=True,
    reference_volume=0)

# DTI Fitting
fa, md = fit_dti(preproc_dwi, preproc_affine, mask, gtab, out_dir=output_dir)

# # Coregistration of a WM mask
# coregister_mask(moving_file, fixed_file, output_dir)