# Fine-Grained Recognition Using Pretrained Self-Supervised Models

## Development

```bash
conda env create -f environment.yml -q
conda env update -f environment.yml --prune
conda remove -n FGR --all

git submodule add https://github.com/huggingface/pytorch-image-models.git
cd pytorch-image-models
git checkout v1.0.15
pip install -e .

git clone git@github.com:ChuaHanChong/dinov2.git
cd dinov2
git checkout MSAI-Project
pip install -e . --no-deps

git clone git@github.com:RobinWu218/SimDINO.git
```

### Command

```bash
python validate.py --data-dir /imagenet/validation/ --model vit_base_patch16_224 --pretrained
```
