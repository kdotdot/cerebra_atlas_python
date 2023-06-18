CerebrA is an accurate non-linear registration of cortical and subcortical labelling from Mindboggle 101 to the symmetric MNI-ICBM2009c atlas followed by manual editing.

### REQUIREMENTS:

- Python3 + dependencies (TODO)
- Freesurfer (optional)

### BUILDING SURFACES (optional)

Cerebra atlas in the head coordinate frame + details for the labels

- CerebrA_in_head.mgz
- CerebrA_LabelDetails.csv
- ICBM 2009c NLS cortical reconstruction folder (optional) (freesurfer subject): Includes ouputs for recon-all command as well as manually edited BEM surfaces.
- CerebrA (optional) (https://gin.g-node.org/anamanera/CerebrA/src/master/)
- ICBM 2009c Nonlinear Symmetric (optional) (https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/)

![BEM MANUAL EDIT](./images/bem_manual_edit.png)

CerebrA_in_head.mgz can be computed using Freesurfer and the following commands. Transforms volume from Native anatomical space (193, 229, 193) to Freesurfer space (256, 256, 256):

##### Convert original .nii into .mgz

`$ mri_convert $CEREBRA_DIR"/CerebrA.nii" $CEREBRA_DIR"/CerebrA.mgz"`

##### Transform CerebrA volumne into Freesurfer 'head' coordinate frame (256x256x256)

`$ mri_vol2vol --mov $CEREBRA_DIR"/CerebrA.mgz" --o $CEREBRA_DIR"/CerebrA_in_head.mgz" --regheader --targ $SUBJECT_DIR/mri/T1.mgz `

TODOS:

- Create python dependencies file
