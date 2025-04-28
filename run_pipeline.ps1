# run_pipeline.ps1

conda activate lcagcn_osr310

# 1. Closed-set training
python train_closedset.py --dataset MSR --known_classes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
python train_closedset.py --dataset UTK --known_classes 0 1 2 3 4 5 6 7

# 2. Extract features & fit Weibull
python extract_activations_and_fit.py

# 3. OpenMax inference & evaluation
python openmax_inference.py
