# Rain or Pumps? - common tasks.
#   make data     build the training table (no download needed)
#   make train    baselines + transformer
#   make report   attention figure and district residuals
#   make all      the three above, in order
PY ?= python3

.PHONY: all data cube train cv report bilstm-data bilstm geo sim ui-data ui test fetch clean-built

all: data train report

## build the training table from the packed rainfall cube
data:
	$(PY) models/build_training_data.py

## rebuild the packed cube from the IMD NetCDFs (only after `make fetch`)
cube:
	$(PY) scripts/build_rain_cube.py

## baselines, transformer and the sequence-only ablation
train:
	$(PY) models/train_transformer.py --epochs 25

## five district folds: does it transfer to districts it has never seen
cv:
	$(PY) models/train_transformer.py --cv --epochs 20

## attention figure and the district residual table
report:
	$(PY) models/attention_report.py

## the BiLSTM copy of the training data (bilstm-data/out/)
bilstm-data:
	$(PY) bilstm-data/build_bilstm_data.py

## BiLSTM training, locally; the full run goes to Kaggle with `python kaggle/push.py`
bilstm:
	cd models && $(PY) train_bilstm.py --epochs 25

## simulator: map layers (needs data/geo/raw/, see simulator/README.md)
geo:
	$(PY) simulator/geo/build_geo.py

## simulator: train locally (the reported run is on Kaggle: python kaggle/push.py --script run_sim.py)
sim:
	$(PY) simulator/train_sim.py --data bilstm-data/out --out simulator/out/sim_6ch

## simulator: everything the UI reads, from a trained run
ui-data:
	$(PY) simulator/ui/build_ui_data.py --run simulator/artifacts

## simulator: serve the UI at http://localhost:8765
ui:
	$(PY) -m http.server 8765 -d simulator/ui

## the pipeline's own test suite
test:
	cd pipeline && pytest -q

## download the raw inputs (only needed for the full 32,299-well rebuild)
fetch:
	$(PY) scripts/fetch_data.py --all

## remove everything that can be rebuilt
clean-built:
	rm -rf data/training data/processed data_cleaning/data bilstm-data/out bilstm-data/kaggle_stage simulator/out
