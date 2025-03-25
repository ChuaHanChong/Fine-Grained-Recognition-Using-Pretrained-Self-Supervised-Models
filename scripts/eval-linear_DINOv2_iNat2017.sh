#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/iNat2017/linear-probing/ViT-B-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 2262 \
#--eval-period-iterations 2262 \
#--train-dataset INat_2017:split=TRAIN:root=/data/hanchong/open-source-data/iNat2017/processed-data:extra=/data/hanchong/open-source-data/iNat2017/processed-data \
#--val-dataset INat_2017:split=VAL:root=/data/hanchong/open-source-data/iNat2017/processed-data:extra=/data/hanchong/open-source-data/iNat2017/processed-data

#CUDA_VISIBLE_DEVICES=3 python eval_linear.py \
#--config-file ./configs/ViT-L-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitl16_reg4_SimDINOv2_100ep.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/iNat2017/linear-probing/ViT-L-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 2262 \
#--eval-period-iterations 2262 \
#--train-dataset INat_2017:split=TRAIN:root=/data/hanchong/open-source-data/iNat2017/processed-data:extra=/data/hanchong/open-source-data/iNat2017/processed-data \
#--val-dataset INat_2017:split=VAL:root=/data/hanchong/open-source-data/iNat2017/processed-data:extra=/data/hanchong/open-source-data/iNat2017/processed-data
