python3 -m venv venvs
. /work/fMRI_analysis/venvs/bin/activate
python3 -m pip install ipykernel
python3 -m ipykernel install --user --name=venvs
bash install_python_packages.sh
echo Done!