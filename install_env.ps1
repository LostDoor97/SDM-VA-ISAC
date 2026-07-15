# SDM-VA-ISAC — Installation de l'environnement Sionna RT
# Exécuter depuis le dossier PFA avec : .\install_env.ps1
# Durée estimée : 10-20 min selon la connexion (téléchargement ~5-6 GB)

$PYTHON = "C:\Users\AdminFix\Desktop\PFA\.venv\Scripts\python.exe"
$PFA    = "C:\Users\AdminFix\Desktop\PFA"

Write-Host "=== [1/5] PyTorch + CUDA 12.4 ===" -ForegroundColor Cyan
uv pip install --python $PYTHON torch --index-url https://download.pytorch.org/whl/cu124

Write-Host "=== [2/5] Mitsuba 3.8.0 + DrJIT 1.3.1 ===" -ForegroundColor Cyan
uv pip install --python $PYTHON "mitsuba==3.8.0" "drjit==1.3.1"

Write-Host "=== [3/5] sionna-rt (local) ===" -ForegroundColor Cyan
uv pip install --python $PYTHON -e "$PFA\sionna\ext\sionna-rt"

Write-Host "=== [4/5] sionna (local, sans sionna.rt) ===" -ForegroundColor Cyan
uv pip install --python $PYTHON -e "$PFA\sionna" --no-deps
# Dépendances manuelles (sionna-rt déjà installé, torch déjà installé)
uv pip install --python $PYTHON h5py importlib-resources scipy matplotlib numpy

Write-Host "=== [5/5] Jupyter + outils notebook ===" -ForegroundColor Cyan
uv pip install --python $PYTHON jupyter ipykernel ipywidgets notebook

# Enregistrer le kernel Jupyter
& $PYTHON -m ipykernel install --user --name "pfa-sionna" --display-name "PFA (Python 3.11)"

Write-Host ""
Write-Host "=== Installation terminée ===" -ForegroundColor Green
Write-Host "Vérification :" -ForegroundColor Yellow
& $PYTHON -c "import torch; print(f'  torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
& $PYTHON -c "import mitsuba as mi; mi.set_variant('cuda_ad_mono_polarized'); print(f'  mitsuba {mi.__version__} | variant: {mi.variant()}')"
& $PYTHON -c "import sionna; print(f'  sionna {sionna.__version__}')"
& $PYTHON -c "import sionna.rt; print(f'  sionna.rt {sionna.rt.__version__}')"

Write-Host ""
Write-Host "Lancer le notebook :" -ForegroundColor Yellow
Write-Host "  cd C:\Users\AdminFix\Desktop\PFA"
Write-Host "  .venv\Scripts\jupyter notebook sionna_rt_osm_pipeline.ipynb"
