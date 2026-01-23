
## TFWaveFormer: Temporal-Frequency Collaborative Multi-level Wavelet Transformer for Dynamic Link Prediction

Most of the used original dynamic graph datasets including Wikipedia, Reddit, MOOC, LastFM, Enron, Social Evo., UCI, Flights, UNtrade and Contact come from (https://zenodo.org/record/7213796#.Y1cO6y8r30o), 
which can be downloaded [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o). 
Please download them and put them in ```DG_data``` folder. 
We can run ```preprocess_data/preprocess_data.py``` for pre-processing the datasets.
For example, to preprocess the *Wikipedia* dataset, we can run the following commands:
```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name wikipedia
```
```{bash}
cd preprocess_data/
python preprocess_all_data.py
```


We've pre-configured the run code in ```run.sh``` for one-click training. Alternatively, you can run:

```python train_link_prediction.py --dataset_name wikipedia --model_name TFWaveFormer --patch_size 1 --max_input_sequence_length 64 --num_runs 5 --gpu 0``` 
or 
```python train_link_prediction.py --dataset_name uci --model_name TFWaveFormer --load_best_configs --num_runs 5 --gpu 0```  for different datasets.

We support dynamic link prediction in both conductive and inductive settings, and employ three negative sampling strategies: random, historical, and inductive.

To save time, we've stored the trained weights in the ```saved_models/TFWaveFormer``` directory; you only need to load them later.

We've pre-configured the run code in ```run.sh``` for one-click testing. Alternatively, you can run:

```python evaluate_link_prediction.py --dataset_name wikipedia --model_name TFWaveFormer --patch_size 1 --max_input_sequence_length 64 --num_runs 1 --gpu 0 --negative_sample_strategy random```  
or 
```python evaluate_link_prediction.py --dataset_name uci --model_name TFWaveFormer --load_best_configs --num_runs 1 --gpu 0 --negative_sample_strategy random```  for different datasets



If you want to load the best model configurations determined by the grid search, please set the *load_best_configs* argument to True.
