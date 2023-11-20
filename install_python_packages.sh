. /fMRI_analysis/venvs/bin/activate
python -m pip install imageio
python -m pip install scipy
python -m pip install numpy
python -m pip install matplotlib==3.7.1 # Some MNE plotting does not work with 3.7.2. Should be fixed with 3.7.3
python -m pip install dicom2nifti
python -m pip install nibabel
python -m pip install nilearn
python -m pip install scikit-learn
python -m pip install mne
python -m pip install seaborn
python -m pip install crtoolbox
python -m pip install ipywidgets
python -m pip install pandas
#python -m pip install pyvistaqt
python -m pip install itertools
python -m pip install ndslib
python -m pip install pickle
echo Done!