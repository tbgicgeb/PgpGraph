# PgpGraph

PgpGraph is based on Graph Convolution Networks, which classify P-glycoprotein inhibitors vs. non-inhibitors and substrates vs. non-substrates.

![pgpgraph_main2(1)](https://github.com/user-attachments/assets/06a55935-c963-4369-ab10-8c96d6bfb27c)


### Dependencies Required
-----------------
The repository has been tested on Ubuntu 22.04.5 LTS. Below is the list of packages required to run PgpGraph: 

``` Numpy: 1.26.4 ```

``` Pandas: 1.5.3 ```

``` logging: 0.5.1.2 ```

``` matplotlib: 3.8.2 ```

``` rdkit: 2022.09.5 ```

``` torch: 2.2.0+cu121 ```

``` mxnet: 1.9.1 ```

``` torchmetrics: 1.3.0.post0 ```

``` torch_geometric: 2.5.0 ```


### Key Files
----------------------
PgpGraph leverages MolGraphConvFeaturizer from the DeepChem Package. To use PgpGraph, first download the required files:

(1) **deepchem_utils.py** (To get the features for your input SMILES)

(2) **pgp_inhibitor_model_weights.pth** (Inhibitor Model)

(3) **pgp_substrate_model_weights.pth** (Substrate Model)

(4) **pgp_inhibitors_test.py** (The script for Inhibitor Model)

(5) **pgp_substrate_test.py** (The script for Substrate Model)

Keep all the required files in the same directory.


### Code to run the script
----------------------

##### (1) For Inhibitors:

   ```bash
   python3 /path/to/pgp_inhibitor_test.py --input_csv /path/to/inhibitor_examples.csv --model_weights /path/to/pgp_inhibitor_model_weights.pth --results_csv /path/to/inhibitors_results/inhibitor_example_results.csv --output_dir /path/to/inhibitors_results > ./output.log &
```

##### (2) For Substrates:

```bash 
python3 /path/to/pgp_substrate_test.py --input_csv /path/to/substrate_examples.csv --model_weights /path/to/pgp_substrate_model_weights.pth --results_csv /path/to/substrates_results/substrate_example_results.csv --output_dir /path/to/substrates_results > ./output.log &
```


#### Caution
--------
Input molecules must be in SMILES format in a .csv file. The progress of the code can be traced using the **output.log** file.


#### Output
-------
The prediction results will be saved in **inhibitor_example_results.csv** for inhibitors and **substrate_example_results.csv** for substrates. Within the substrate_results and inhibitors_results directories, the feature importance plots for inhibitors and substrates will be saved in inhibitors_barplots or substrates_barplots directories, respectively. 
