import os
import numpy as np
from dipy.io.image import load_nifti
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import connectivity_matrix, seeds_from_mask
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
# from dipy.data import fetch_atlas_harvard_oxford, read_atlas_harvard_oxford
from dipy.data import fetch_mni152, read_mni152_template


class CustomTensorDirectionGetter:
    """
    Custom Direction Getter using the principal eigenvectors from the tensor model.
    """
    def __init__(self, principal_directions, mask):
        # Ensure writable and contiguous arrays
        self.principal_directions = np.ascontiguousarray(principal_directions)
        self.mask = np.ascontiguousarray(mask)

    def initial_direction(self, point):
        """
        Returns the initial direction at the given seed point.
        """
        return self._get_direction(point)

    def get_direction(self, point, _):
        """
        Returns the direction at the given point (ignoring streamline context).
        """
        return self._get_direction(point)

    def _get_direction(self, point):
        """
        Retrieve the direction from the principal eigenvectors.
        """
        i, j, k = np.round(point).astype(int)
        if (0 <= i < self.principal_directions.shape[0] and
            0 <= j < self.principal_directions.shape[1] and
            0 <= k < self.principal_directions.shape[2]):
            if self.mask[i, j, k]:
                return self.principal_directions[i, j, k]
        return None


def adjacency_matrix(dwi_file, mask_file, atlas_file, bval_file, bvec_file, output_dir):
    """
    Calculate the adjacency matrix from DWI data using a reference atlas.

    Parameters
    ----------
    dwi_file : str
        Path to the preprocessed DWI file.
    mask_file : str
        Path to the brain mask file.
    atlas_file : str
        Path to the reference atlas file (aligned to DWI space).
    bval_file : str
        Path to the b-values file.
    bvec_file : str
        Path to the b-vectors file.
    output_dir : str
        Directory where the adjacency matrix will be saved.

    Returns
    -------
    adjacency_matrix : np.ndarray
        Adjacency matrix based on the atlas regions.
    labels : np.ndarray
        List of region labels from the atlas.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load preprocessed DWI data and mask
    dwi, affine = load_nifti(dwi_file)
    mask, _ = load_nifti(mask_file)

    # Ensure mask is writable and contiguous
    mask = np.ascontiguousarray(mask)

    # Load atlas
    atlas, _ = load_nifti(atlas_file)

    # Load gradient table
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file).T
    gtab = gradient_table(bvals, bvecs)

    # Fit DTI model
    print("Fitting DTI model...")
    dti_model = TensorModel(gtab)
    dti_fit = dti_model.fit(dwi, mask=mask)

    # Generate stopping criterion based on FA
    fa = np.ascontiguousarray(dti_fit.fa)  # Ensure FA is writable
    stopping_criterion = BinaryStoppingCriterion(fa > 0.2)

    # Create seeds from the mask
    seeds = seeds_from_mask(mask, density=1, affine=affine)

    # Use principal eigenvectors for deterministic tractography
    print("Generating streamlines...")
    principal_directions = np.ascontiguousarray(dti_fit.evecs[..., 0])  # Ensure writable array
    direction_getter = CustomTensorDirectionGetter(principal_directions, mask)

    # Perform deterministic tractography
    streamlines_generator = LocalTracking(
        direction_getter, stopping_criterion, seeds, affine, step_size=0.5
    )
    streamlines = Streamlines(streamlines_generator)

    # Compute adjacency matrix
    print("Computing adjacency matrix...")
    adjacency_matrix, region_labels = connectivity_matrix(
        streamlines,
        atlas,
        affine=affine,
        return_mapping=False,
        mapping_as_streamlines=False,
        symmetric=True
    )

    # Save adjacency matrix
    adjacency_matrix_file = os.path.join(output_dir, "adjacency_matrix.npy")
    np.save(adjacency_matrix_file, adjacency_matrix)
    print(f"Adjacency matrix saved to: {adjacency_matrix_file}")

    return adjacency_matrix, region_labels


def adjacency_matrix_with_harvard_oxford(
    dwi_file, mask_file, bval_file, bvec_file, output_dir
):
    """
    Calculate the adjacency matrix using the Harvard-Oxford atlas.

    Parameters
    ----------
    dwi_file : str
        Path to the preprocessed DWI file.
    mask_file : str
        Path to the brain mask file.
    bval_file : str
        Path to the b-values file.
    bvec_file : str
        Path to the b-vectors file.
    output_dir : str
        Directory where the adjacency matrix will be saved.

    Returns
    -------
    adjacency_matrix : np.ndarray
        Adjacency matrix based on the atlas regions.
    labels : list
        List of region labels from the Harvard-Oxford atlas.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load preprocessed DWI data and mask
    dwi, affine = load_nifti(dwi_file)
    mask, _ = load_nifti(mask_file)

    # Load Harvard-Oxford atlas
    print("Fetching Harvard-Oxford Atlas...")
    fetch_atlas_harvard_oxford()
    atlas, labels = read_atlas_harvard_oxford()

    # Load gradient table
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file).T
    gtab = gradient_table(bvals, bvecs)

    # Fit DTI model
    print("Fitting DTI model...")
    dti_model = TensorModel(gtab)
    dti_fit = dti_model.fit(dwi, mask=mask)

    # Generate stopping criterion based on FA
    fa = dti_fit.fa
    stopping_criterion = BinaryStoppingCriterion(fa > 0.2)

    # Create seeds from the mask
    seeds = seeds_from_mask(mask, density=1, affine=affine)

    # Use principal eigenvectors for deterministic tractography
    print("Generating streamlines...")
    principal_directions = dti_fit.evecs[..., 0]  # Principal eigenvectors

    # Use deterministic direction getter from DIPY
    from dipy.direction import DeterministicDirectionGetter
    from dipy.data import get_sphere

    sphere = get_sphere('repulsion724')
    direction_getter = DeterministicDirectionGetter.from_pmf(
        principal_directions, max_angle=30.0, sphere=sphere
    )

    # Perform deterministic tractography
    streamlines_generator = LocalTracking(
        direction_getter, stopping_criterion, seeds, affine, step_size=0.5
    )
    streamlines = Streamlines(streamlines_generator)

    # Compute adjacency matrix
    print("Computing adjacency matrix...")
    adjacency_matrix, region_labels = connectivity_matrix(
        streamlines,
        atlas,
        affine=affine,
        return_mapping=False,
        mapping_as_streamlines=False,
        symmetric=True
    )

    # Save adjacency matrix
    adjacency_matrix_file = os.path.join(output_dir, "adjacency_matrix.npy")
    np.save(adjacency_matrix_file, adjacency_matrix)
    print(f"Adjacency matrix saved to: {adjacency_matrix_file}")

    return adjacency_matrix, labels


def calculate_adjacency_matrix_with_mni152(
    dwi_file, mask_file, bval_file, bvec_file, output_dir
):
    """
    Calculate the adjacency matrix using the MNI152 atlas.

    Parameters
    ----------
    dwi_file : str
        Path to the preprocessed DWI file.
    mask_file : str
        Path to the brain mask file.
    bval_file : str
        Path to the b-values file.
    bvec_file : str
        Path to the b-vectors file.
    output_dir : str
        Directory where the adjacency matrix will be saved.

    Returns
    -------
    adjacency_matrix : np.ndarray
        Adjacency matrix based on the atlas regions.
    labels : list
        List of region labels from the MNI152 atlas.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load preprocessed DWI data and mask
    dwi, affine = load_nifti(dwi_file)
    mask, _ = load_nifti(mask_file)

    # Load MNI152 atlas
    print("Fetching MNI152 Atlas...")
    fetch_mni152()
    atlas = read_mni152_template()

    # Load gradient table
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file).T
    gtab = gradient_table(bvals, bvecs)

    # Fit DTI model
    print("Fitting DTI model...")
    dti_model = TensorModel(gtab)
    dti_fit = dti_model.fit(dwi, mask=mask)

    # Generate stopping criterion based on FA
    fa = dti_fit.fa
    stopping_criterion = BinaryStoppingCriterion(fa > 0.2)

    # Create seeds from the mask
    seeds = seeds_from_mask(mask, density=1, affine=affine)

    # Use principal eigenvectors for deterministic tractography
    print("Generating streamlines...")
    principal_directions = dti_fit.evecs[..., 0]  # Principal eigenvectors

    # Use deterministic direction getter from DIPY
    from dipy.direction import DeterministicDirectionGetter
    from dipy.data import get_sphere

    sphere = get_sphere('repulsion724')
    direction_getter = DeterministicDirectionGetter.from_pmf(
        principal_directions, max_angle=30.0, sphere=sphere
    )

    # Perform deterministic tractography
    streamlines_generator = LocalTracking(
        direction_getter, stopping_criterion, seeds, affine, step_size=0.5
    )
    streamlines = Streamlines(streamlines_generator)

    # Compute adjacency matrix
    print("Computing adjacency matrix...")
    adjacency_matrix, region_labels = connectivity_matrix(
        streamlines,
        atlas,
        affine=affine,
        return_mapping=False,
        mapping_as_streamlines=False,
        symmetric=True
    )

    # Save adjacency matrix
    adjacency_matrix_file = os.path.join(output_dir, "adjacency_matrix.npy")
    np.save(adjacency_matrix_file, adjacency_matrix)
    print(f"Adjacency matrix saved to: {adjacency_matrix_file}")

    return adjacency_matrix, region_labels
