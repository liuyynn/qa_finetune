# 环境安装
```
conda create -n qa_env python=3.9 -y
conda activate qa_env

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

pip install transformers==4.36.2 datasets==2.14.6
pip install scikit-learn pandas tqdm
pip install accelerate==0.23.0
```

# 项目结构
```
qa_finetune
├── README.md
├── data                # 数据集目录
│   └── sciq
│       ├── processed   # 预处理后的数据
│       └── raw         # 原始数据
├── models
│   └── roberta-base
├── outputs
│   └── roberta-sciq
│       ├── best_model          # 微调模型文件
│       └── test_preds.jsonl    # 预测结果文件
├── predict_roberta.py  # 模型预测
└── train_roberta.py    # 模型训练
```
[robrtta-base 模型下载](https://huggingface.co/roberta-base)

---
# 示例
```
# sciq-roberta 模型训练 
python train_roberta.py --data_dir data/sciq/processed --model_dir models/roberta-base --output_dir outputs/roberta-sciq

# sciq-test 预测
python predict_roberta.py --model_dir outputs/roberta-sciq/best_model --test_path data/sciq/processed/test.jsonl --output_path outputs/roberta-sciq/test_preds.jsonl
```