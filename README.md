# Inductive Spatio-Temporal Kriging with Heterogeneous Relations

## Structure

* METR-LA: includes the source code and data of METR-LA dataset
* Beijing: includes the source code and data of Beijing dataset

Due to confidentiality, we currently do not provide the code and data on Xiamen dataset

## Training

To train ISTK-HR(SP) on the METR-LA dataset, cd to the folder "METR-LA/ISTK-HR(SP)", and run "python train.py"

To train ISTK-HR(SP) on the Beijing dataset, cd to the folder "Beijing/ISTK-HR(SP)", and run "python train.py"

To train ISTK-HR(SP+FS) on the Beijing dataset, cd to the folder "Beijing/ISTK-HR(SP+FS)", and run "python train.py"

## Testing

We provide pre-trained model files on both datasets.

To evaluate ISTK-HR(SP) on the METR-LA dataset, cd to the folder "METR-LA/ISTK-HR(SP)", and run "python test.py"

To evaluate ISTK-HR(SP) on the Beijing dataset, cd to the folder "Beijing/ISTK-HR(SP)", and run "python test.py"

To evaluate ISTK-HR(SP+FS) on the Beijing dataset, cd to the folder "Beijing/ISTK-HR(SP+FS)", and run "python test.py"

## Results

We provide pre-trained models on both datasets, which achieve the following performance:

|    METR-LA     |  RMSE  |   MAE  |  MAPE  |   R2  |
| -------------- | ------ | ------ | ------ | ----- |
| ISTK-HR(SP)    | 8.602  | 5.425  | 0.137  | 0.844 |


|    Beijing     |  RMSE  |   MAE  |  MAPE  |   R2  |
| -------------- | ------ | ------ | ------ | ----- |
| ISTK-HR(SP)    | 34.552 | 19.584 | 0.247  | 0.831 |
| ISTK-HR(SP+FS) | 34.001 | 18.851 | 0.239  | 0.836 |

