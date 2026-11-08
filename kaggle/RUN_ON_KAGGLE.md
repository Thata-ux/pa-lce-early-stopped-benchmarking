# Running the pipeline on Kaggle (CPU, free tier)

Settings: Accelerator **None**, Internet **on**. Run each session with
**Save Version -> Save & Run All (Commit)** so it continues when the browser is closed
(limit: 12 hours per session). All stages skip work that already exists, so an
interrupted session can be continued by adding its output as input to a new notebook.

## Session A - generate and LC-PFN (can be split by `--dims`)

```
!pip install -q coco-experiment==2.8.3 cma
!git clone -q https://github.com/automl/lcpfn.git
!python pilot_lce_final.py generate --dims 20
!python pilot_lce_final.py generate --dims 10
!python pilot_lce_final.py generate --dims 5
!python pilot_lce_final.py lcpfn --lcpfn_repo lcpfn
```

## Preparation cell for later sessions

Inputs: the dataset holding `pilot_lce_final.py` and the output of the notebook that
holds `results_final/` (and `lcpfn/`).

```python
import os, shutil, glob

def find(name, folder=False):
    out = []
    for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
        if (folder and name in dirs) or (not folder and name in files):
            out.append(os.path.join(root, name))
    return out

shutil.copy(find('pilot_lce_final.py')[0], '/kaggle/working/pilot_lce_final.py')
rf = [p for p in find('results_final', folder=True) if os.path.isdir(os.path.join(p, 'parts'))]
shutil.copytree(rf[0], '/kaggle/working/results_final', dirs_exist_ok=True)
lc = find('lcpfn', folder=True)
if lc:
    shutil.copytree(lc[0], '/kaggle/working/lcpfn', dirs_exist_ok=True)
for d in (5, 10, 20):
    print(f'd{d}:', len(glob.glob(f'/kaggle/working/results_final/parts/d{d}/*.npz')), 'part files')
```

## Session B - main analyses (about 4 hours)

```
!python pilot_lce_final.py analyze --tag v3
!python pilot_lce_final.py extras --tag v3
!python pilot_lce_final.py analyze --cv train1 --tag v3
```

## Session C - unseen algorithms (about 6-8 hours, can run in parallel with B)

```
!python pilot_lce_final.py analyze --cv loao_fold6 --scenarios within --tag v3
```

## Session D - extension with CMA-ES and SHADE (about 3-3.5 hours)

```
!pip install -q coco-experiment==2.8.3 cma
A="PSO-w PSO-cf DE-rand1 DE-best1 DE-ctb1 GWO WOA HHO CMA-ES SHADE"
!python pilot_lce_final.py generate --algs $A     # 144 new combinations only
!python pilot_lce_final.py lcpfn --algs $A --lcpfn_repo lcpfn
!python pilot_lce_final.py analyze --algs $A --tag ext10
```

Recording the environment: add `!pip freeze > results_final/pip_freeze.txt` to every session.
