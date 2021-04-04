# Scripts used to run the expriments in the paper 

## Download the data 

We process the raw count data with cell labels downloaded from scanorama and saved the processed data with the IHPF components in the AnnData format. These processed data corresponds to datasets for Scenario  A,B,C,D in the paper. 

To rerun the data generation process, the raw count data needs to be stored in the folder /Real/scanorama/ and cell labels needs to be stored in /Real/scanorama/cell_labels/. These datasets can be downloaded from scanorama https://github.com/brianhie/scanorama. 

## Running the experiments 

main.py stores the parameters and file locations to run the grid search for hyper-parameters and cell clustering of data integration. 

Different options such as hyper, benchmark, integration_geneselection, integration_hyper corresponds to results in different sections in the paper.


## Visualisation 

The visualisation is done with Matplotlib and seaborn in Python notebooks. 



