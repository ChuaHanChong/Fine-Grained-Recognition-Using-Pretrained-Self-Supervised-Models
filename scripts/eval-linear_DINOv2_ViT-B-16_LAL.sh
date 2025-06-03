#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Birdsnap/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 136 \
#--eval-period-iterations 136 \
#--logit-adjusted-loss \
#--train-dataset Birdsnap:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/Birdsnap/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Birdsnap/processed-data \
#--val-dataset Birdsnap:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/Birdsnap/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Birdsnap/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Country211/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 123 \
#--eval-period-iterations 123 \
#--logit-adjusted-loss \
#--train-dataset Country211:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/Country211/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Country211/processed-data \
#--val-dataset Country211:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/Country211/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Country211/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/CUB-200-2011/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 21 \
#--eval-period-iterations 21 \
#--logit-adjusted-loss \
#--train-dataset CUB200_2011:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/CUB-200-2011/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/CUB-200-2011/processed-data \
#--val-dataset CUB200_2011:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/CUB-200-2011/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/CUB-200-2011/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/FGVC_Aircraft/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 13 \
#--eval-period-iterations 13 \
#--logit-adjusted-loss \
#--train-dataset FGVC_Aircraft:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/FGVC_Aircraft/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/FGVC_Aircraft/processed-data \
#--val-dataset FGVC_Aircraft:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/FGVC_Aircraft/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/FGVC_Aircraft/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Food-101/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 291 \
#--eval-period-iterations 291 \
#--logit-adjusted-loss \
#--train-dataset Food101:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/Food-101/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Food-101/processed-data \
#--val-dataset Food101:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/Food-101/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Food-101/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/NABirds/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 82 \
#--eval-period-iterations 82 \
#--logit-adjusted-loss \
#--train-dataset NABirds:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/NABirds/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/NABirds/processed-data \
#--val-dataset NABirds:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/NABirds/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/NABirds/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Oxford_Flowers/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 64 \
#--epoch-length 15 \
#--eval-period-iterations 15 \
#--logit-adjusted-loss \
#--train-dataset OxfordFlowers:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/Oxford_Flowers/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Oxford_Flowers/processed-data \
#--val-dataset OxfordFlowers:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/Oxford_Flowers/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Oxford_Flowers/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Oxford_Pets/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 13 \
#--eval-period-iterations 13 \
#--logit-adjusted-loss \
#--train-dataset OxfordPets:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/Oxford_Pets/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Oxford_Pets/processed-data \
#--val-dataset OxfordPets:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/Oxford_Pets/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Oxford_Pets/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/RESISC45/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 105 \
#--eval-period-iterations 105 \
#--logit-adjusted-loss \
#--train-dataset RESISC45:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/RESISC45/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/RESISC45/processed-data \
#--val-dataset RESISC45:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/RESISC45/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/RESISC45/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Stanford_Cars/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 27 \
#--eval-period-iterations 27 \
#--logit-adjusted-loss \
#--train-dataset StanfordCars:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/Stanford_Cars/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Stanford_Cars/processed-data \
#--val-dataset StanfordCars:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/Stanford_Cars/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Stanford_Cars/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/Stanford_Dogs/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 44 \
#--eval-period-iterations 44 \
#--logit-adjusted-loss \
#--train-dataset StanfordDogs:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/Stanford_Dogs/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Stanford_Dogs/processed-data \
#--val-dataset StanfordDogs:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/Stanford_Dogs/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/Stanford_Dogs/processed-data

#CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
#--config-file ./configs/ViT-B-16_DINOv2.yml \
#--pretrained-weights /data/hanchong/open-source-weights/processed-weights/vitb16_reg4_SimDINOv2_ep100.pth \
#--output-dir /data/hanchong/open-source-weights/trained-weights/SUN397/linear-probing/ViT-B-16_DINOv2_LAL \
#--epochs 50 \
#--save-checkpoint-frequency 1 \
#--batch-size 256 \
#--epoch-length 341 \
#--eval-period-iterations 341 \
#--logit-adjusted-loss \
#--train-dataset SUN397:split=TRAIN:root=/data/hanchong/open-source-data/PrimaryDatasets/SUN397/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/SUN397/processed-data \
#--val-dataset SUN397:split=VAL:root=/data/hanchong/open-source-data/PrimaryDatasets/SUN397/processed-data:extra=/data/hanchong/open-source-data/PrimaryDatasets/SUN397/processed-data
