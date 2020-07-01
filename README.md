# ReferCam

Refer to [https://github.com/zyang-ur/onestage_grounding](https://github.com/zyang-ur/onestage_grounding)

python3 train_yolo.py --data_root ./ln_data/ --dataset referit   --gpu 0 --savename="ReferCam"


python3 train_yolo.py --data_root ./ln_data/ --dataset unc+ --gpu 1 --savename="ReferCam" --save_plot --batch_size 36 --lr 2e-4 --resume saved_models/ReferCam_model_best.pth.tar
