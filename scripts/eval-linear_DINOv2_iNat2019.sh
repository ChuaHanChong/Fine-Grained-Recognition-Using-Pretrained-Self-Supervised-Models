#CUDA_VISIBLE_DEVICES=2 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/iNat2019/linear-probing/ViT-B-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 1035 \
#--eval-period-iterations 1035 \
#--train-dataset INat_2019:split=TRAIN:root=/data/hanchong/open-source-data/iNat2019/processed-data:extra=/data/hanchong/open-source-data/iNat2019/processed-data \
#--val-dataset INat_2019:split=VAL:root=/data/hanchong/open-source-data/iNat2019/processed-data:extra=/data/hanchong/open-source-data/iNat2019/processed-data

#CUDA_VISIBLE_DEVICES=2 python eval_linear.py \
#--config-file ./configs/ViT-L-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitl16_reg4_SimDINOv2_100ep.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/iNat2019/linear-probing/ViT-L-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 1035 \
#--eval-period-iterations 1035 \
#--train-dataset INat_2019:split=TRAIN:root=/data/hanchong/open-source-data/iNat2019/processed-data:extra=/data/hanchong/open-source-data/iNat2019/processed-data \
#--val-dataset INat_2019:split=VAL:root=/data/hanchong/open-source-data/iNat2019/processed-data:extra=/data/hanchong/open-source-data/iNat2019/processed-data
