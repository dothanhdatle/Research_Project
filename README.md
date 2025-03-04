# Research Project: Self-Supervised learning for Human Gestures Recognition

The aim of the project is constructing a framework for self-supervised learning for human gestures recognition on skeleton-based dataset SHREC22.

In this project, a contrastive learning framework based on MoCo v3 was built using ST-GCN as query and key encoders to encode the representations for input skeleton sequences. The temperature parameter can be fixed or dynamically scaled for each key, query pair.

The experiment on SHREC22 dataset is in the notebook training_moco_v3.

# Main reference:
- ST-GCN: https://arxiv.org/abs/1801.07455
- MoCo: https://arxiv.org/abs/1911.05722
- MoCo v2: https://arxiv.org/abs/2003.04297v1
- MoCo v3: https://arxiv.org/abs/2104.02057
- Dynamically Scaled Temperature: https://arxiv.org/abs/2308.01140
