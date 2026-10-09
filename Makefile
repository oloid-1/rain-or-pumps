# Rain or Pumps? - common tasks.
#
# Run the app (its data is committed):
#   pip install -r requirements.txt
#   make api               http://localhost:8000
#
# Rebuild the app's data from data/ (needs requirements-train.txt, ~20 min):
#   make app-data
PY ?= python3

.PHONY: data sequences cube train cv report bilstm sim geo places ui-data forecast app-data api test fetch clean-built

## the training table and the 1-channel rain sequence (data/training/)
data:
	$(PY) models/build_training_data.py

## the training table plus the BiLSTM's 6-channel sequence (runs `data` itself)
sequences:
	$(PY) models/build_sequences.py

## rebuild the packed rain cube from the IMD NetCDFs (only after `make fetch`)
cube:
	$(PY) scripts/build_rain_cube.py

## comparison models: baselines + transformer, district folds, attention figure, BiLSTM
train:
	$(PY) models/train_transformer.py --epochs 25
cv:
	$(PY) models/train_transformer.py --cv --epochs 20
report:
	$(PY) models/attention_report.py
bilstm:
	cd models && $(PY) train_bilstm.py --epochs 25

## the chosen model, trained locally (the reported run is on Kaggle: python kaggle/push.py --script run_sim.py)
sim:
	$(PY) simulator/train_sim.py --out simulator/out/sim_6ch

## map layers and the town list for search (need downloads in data/geo/raw/, see simulator/README.md)
geo:
	$(PY) simulator/geo/build_geo.py
places:
	$(PY) simulator/geo/build_places.py

## everything the UI reads, then the forecast table (~18 min on CPU for the full data)
ui-data:
	$(PY) simulator/ui/build_ui_data.py --run simulator/artifacts
forecast:
	$(PY) simulator/forecast/build_forecast.py

## the API with the UI at http://localhost:8000 (docs at /docs)
api:
	$(PY) -m uvicorn main:app --port 8000

test:
	$(PY) -m pytest simulator/api -q

## all of the app's data: sequences, ui-data and forecast in one go
app-data:
	$(PY) scripts/build_app.py

## download the raw inputs
fetch:
	$(PY) scripts/fetch_data.py --all

## remove everything that can be rebuilt
clean-built:
	rm -rf data/training data/kaggle data/kaggle_stage simulator/out
