#CUDA_VISIBLE_DEVICES=2 python eval_linear.py \
#--config-file ./configs/ViT-L-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitl16_reg4_SimDINOv2_100ep.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/NABirds/linear-probing/ViT-L-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 64 \
#--epoch-length 331 \
#--eval-period-iterations 331 \
#--train-dataset NABirds:split=TRAIN:root=/data/hanchong/open-source-data/NABirds/processed-data:extra=/data/hanchong/open-source-data/NABirds/processed-data \
#--val-dataset NABirds:split=VAL:root=/data/hanchong/open-source-data/NABirds/processed-data:extra=/data/hanchong/open-source-data/NABirds/processed-data

#CUDA_VISIBLE_DEVICES=2 python eval_linear.py \
#--config-file ./configs/ViT-L-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitl16_reg4_SimDINOv2_100ep.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Birdsnap/linear-probing/ViT-L-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 64 \
#--epoch-length 546 \
#--eval-period-iterations 546 \
#--train-dataset Birdsnap:split=TRAIN:root=/data/hanchong/open-source-data/Birdsnap/processed-data:extra=/data/hanchong/open-source-data/Birdsnap/processed-data \
#--val-dataset Birdsnap:split=VAL:root=/data/hanchong/open-source-data/Birdsnap/processed-data:extra=/data/hanchong/open-source-data/Birdsnap/processed-data

#CUDA_VISIBLE_DEVICES=2 python eval_linear.py \
#--config-file ./configs/ViT-L-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitl16_reg4_SimDINOv2_100ep.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/FGVC_Aircraft/linear-probing/ViT-L-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 32 \
#--epoch-length 104 \
#--eval-period-iterations 104 \
#--train-dataset FGVC_Aircraft:split=TRAIN:root=/data/hanchong/open-source-data/FGVC_Aircraft/processed-data:extra=/data/hanchong/open-source-data/FGVC_Aircraft/processed-data \
#--val-dataset FGVC_Aircraft:split=VAL:root=/data/hanchong/open-source-data/FGVC_Aircraft/processed-data:extra=/data/hanchong/open-source-data/FGVC_Aircraft/processed-data

#CUDA_VISIBLE_DEVICES=2 python eval_linear.py \
#--config-file ./configs/ViT-L-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitl16_reg4_SimDINOv2_100ep.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Stanford_Cars/linear-probing/ViT-L-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 32 \
#--epoch-length 223 \
#--eval-period-iterations 223 \
#--train-dataset StanfordCars:split=TRAIN:root=/data/hanchong/open-source-data/Stanford_Cars/processed-data:extra=/data/hanchong/open-source-data/Stanford_Cars/processed-data \
#--val-dataset StanfordCars:split=VAL:root=/data/hanchong/open-source-data/Stanford_Cars/processed-data:extra=/data/hanchong/open-source-data/Stanford_Cars/processed-data

#CUDA_VISIBLE_DEVICES=2 python eval_linear.py \
#--config-file ./configs/ViT-L-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitl16_reg4_SimDINOv2_100ep.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Stanford_Dogs/linear-probing/ViT-L-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 32 \
#--epoch-length 356 \
#--eval-period-iterations 356 \
#--train-dataset StanfordDogs:split=TRAIN:root=/data/hanchong/open-source-data/Stanford_Dogs/processed-data:extra=/data/hanchong/open-source-data/Stanford_Dogs/processed-data \
#--val-dataset StanfordDogs:split=VAL:root=/data/hanchong/open-source-data/Stanford_Dogs/processed-data:extra=/data/hanchong/open-source-data/Stanford_Dogs/processed-data

#CUDA_VISIBLE_DEVICES=2 python eval_linear.py \
#--config-file ./configs/ViT-L-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitl16_reg4_SimDINOv2_100ep.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/CUB200-2011/linear-probing/ViT-L-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 32 \
#--epoch-length 156 \
#--eval-period-iterations 156 \
#--train-dataset CUB200_2011:split=TRAIN:root=/data/hanchong/open-source-data/CUB200-2011/processed-data:extra=/data/hanchong/open-source-data/CUB200-2011/processed-data \
#--val-dataset CUB200_2011:split=VAL:root=/data/hanchong/open-source-data/CUB200-2011/processed-data:extra=/data/hanchong/open-source-data/CUB200-2011/processed-data

#CUDA_VISIBLE_DEVICES=2 python eval_linear.py \
#--config-file ./configs/ViT-L-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitl16_reg4_SimDINOv2_100ep.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Oxford_Flowers/linear-probing/ViT-L-16_DINOv2 \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 16 \
#--epoch-length 63 \
#--eval-period-iterations 63 \
#--train-dataset OxfordFlowers:split=TRAIN:root=/data/hanchong/open-source-data/Oxford_Flowers/processed-data:extra=/data/hanchong/open-source-data/Oxford_Flowers/processed-data \
#--val-dataset OxfordFlowers:split=VAL:root=/data/hanchong/open-source-data/Oxford_Flowers/processed-data:extra=/data/hanchong/open-source-data/Oxford_Flowers/processed-data
