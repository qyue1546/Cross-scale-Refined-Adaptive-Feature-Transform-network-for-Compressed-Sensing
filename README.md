# Cross-scale-Refined-Adaptive-Feature-Transform-network-for-Compressed-Sensing

#Users can download the pre-processed training set from https://drive.google.com/file/d/1F_UpIoOU_kvY_64XihBk6BP_WzwHWOOp/view?usp=drive_link. Training sets and test sets need to be placed under ./DataSets/.

# Test
1. Place the weight file (CSratio_0.X0) in \save_temp. Run the following scripts to test model.

    python test.py  --sensing-rate 0.1 --test_name Set11 --epochs 100

2. You can change the sensing-rate (CS ratio) and test_name (dataset) to get test results for different CS ratios in different datasets. 

3. The `Xnet_params_100.pth` file consists of seven sets of weights obtained by repeatedly running experiments on the same model, with the CS ratio fixed at 0.4 but the random seed left unfixed. If you want to run multiple different models trained at a CS ratio of 0.4, please place the Xnet_params_100.pth file in the \save_temp\CSratio_0.40 folder, remove the first letter from the filename, and then test. 

## Train
1. Run the following scripts to train model.

    python train.py --sensing_rate 0.4 --layer_num 16 --learning_rate 2e-4 --start_epoch 0 --epochs 100 --batch_size 32
