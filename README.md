<H1 style="text-align: center;">Cerebra atlas Python</H1>




<b>cerebra_atlas_python</b> offers Python 

<a href="https://nist.mni.mcgill.ca/cerebra/">CerebrA</a> is an accurate non-linear registration of cortical and subcortical labelling from <a href="https://nist.mni.mcgill.ca/cerebra/">Mindboggle 101</a> to the <a href="https://nist.mni.mcgill.ca/cerebra/">symmetric MNI-ICBM2009c atlas</a>. <b>cerebra_atlas_python</b> abstracts the following:



cerebra_atlas_python functionality

abstract coordinate frame transformations

https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/

<div style="display:flex;align-items:center;justify-content:center;background-color:aliceblue;padding:25px;flex-direction:column"><img src="./images/example.png" alt="BEM MANUAL EDIT" width=50%></img><br/><small>Cerebra Atlas</small></div>

MNIAverage: https://www.dropbox.com/scl/fi/zoff6ihk3711zn6phu2zt/MNIAverage.zip?rlkey=jrg63liehpuhus4suyyfmibuz&dl=0
Cerebra Atlas: https://www.dropbox.com/scl/fi/ivvh2afex6idffmano3qj/10.12751_g-node.be5e62.zip?rlkey=ie3w8lbd5b5377xotsjgnwkrg&dl=0

### REQUIREMENTS:

- Python3 + dependencies (TODO)
- Freesurfer (optional)

### USAGE / USE CASES

```
from cerebra_atlas_python import CerebrA
cerebra = CerebrA()
cerebra.orthoview()
```

<div style="display:flex;align-items:center;justify-content:center;padding:25px;flex-direction:column"><img src="./images/orthoview_example.png" alt="BEM MANUAL EDIT" width=100%></img><br/><small>Cerebra Atlas</small></div>

### INSTALL

#### Building wheels

```
$ git clone https://github.com/kdotdot/cerebra_atlas_python.git
$ cd cerebra_atlas_python
$ pip install -r requirements.txt
$ pip install build
$ python -m build
$ pip install .
```

### COMPUTING BRAIN DATA (optional)

The whole data folder generation (cerebra_data) process is outlined in notebooks/0.0-generate-cerebra-data.ipynb

<ol>
  <li>Use Freesurfer to perform cortical reconstruction and generate Boundary Element Model (BEM) surfaces</li>
  <li>Convert CerebrA volume to T1w scan coordinate frame </li>
  <li>Manually align fiducials(?) </li>
</ol>

#### Data sources:

Original datasets used to build the processed versions of the volumes.

- [CerebrA](https://gin.g-node.org/anamanera/CerebrA/src/master/): $CEREBRA_DIR
- [ICBM 2009c Nonlinear Symmetric [NIFTI]](https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/): $ICBM_DIR

#### Steps:

##### 1) Use Freesurfer to perform cortical reconstruction and generate Boundary Element Model (BEM) surfaces
<ul>
  <li>Perform cortical reconstruction from MRI scan using recon-all</li>

`$ recon-all -subjid icbm152 -i $ICBM_DIR/mni_icbm152_t1_tal_nlin_sym_09c.nii -all`
  <li>Generate boundary element model (BEM) using watershed algoritm</li>
  BEM surfaces were generated using the FreeSurfer watershed algorithm through MNE's `mne.bem.make_watershed_bem`
  <li>Manually edit BEM surfaces </li>
  BEM surfaces are manually edited so that all inner surfaces are contained within the outer surfaces as explained [here](https://mne.tools/stable/auto_tutorials/forward/80_fix_bem_in_blender.html).
  <div style="display:flex;align-items:center;justify-content:center;background-color:aliceblue;padding:25px;flex-direction:column"><img src="./images/bem_manual_edit.png" alt="BEM MANUAL EDIT" width=50%></img><br/><small>Manual editing of BEM surfaces produced by recon-all</small></div>
</ul>

##### 2) Convert CerebrA volume to T1w scan coordinate frame


###### Convert original .nii into .mgz

CerebrA_in_head.mgz can be computed using Freesurfer and the following commands. Transforms volume from Native anatomical space (193, 229, 193) to Freesurfer space (256, 256, 256):

`$ mri_convert $CEREBRA_DIR"/CerebrA.nii" $CEREBRA_DIR"/CerebrA.mgz"`

###### Transform CerebrA volumne into Freesurfer 'head' (head or mri??) coordinate frame (256x256x256)

`$ mri_vol2vol --mov $CEREBRA_DIR"/CerebrA.mgz" --o $CEREBRA_DIR"/CerebrA_in_head.mgz" --regheader --targ $SUBJECT_DIR/mri/T1.mgz `

### TODOS:

- Update README
- Create documentation page
- Publish library
  -> Get DOI
  -> Create python dependencies file
  -> Upload to pip
