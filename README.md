# Federated Domain Generalization with Label Smoothing and Balanced Decentralized Training (FedSB)
![Overview v3](https://github.com/user-attachments/assets/c42f211d-024f-409b-8a17-89f6f53baf47)

This repository contains the implementation of **FedSB**, a novel approach to federated domain generalization that addresses the challenges of data heterogeneity through:
1. **Label Smoothing**: Mitigates overconfidence of local models, enhancing domain generalization.
2. **Balanced Decentralized Training**: Introduces a budgeting mechanism to ensure consistent contributions from all clients.


## Abstract
FedSB utilizes label smoothing at the client level to prevent overfitting to domain-specific features and employs a decentralized budgeting mechanism to balance training contributions. Our experiments on four multi-domain datasets (PACS, VLCS, OfficeHome, and TerraIncognita) demonstrate state-of-the-art performance on three of the four datasets.

## Key Features
- Enhanced domain generalization through label smoothing.
- Balanced contributions from clients with varying data sizes.
- Outperforms competing methods on major benchmarks.

## Datasets Used
- **PACS**
- **OfficeHome**
- **VLCS**
- **TerraIncognita**

## Results
FedSB achieves state-of-the-art results on three out of four datasets, significantly advancing federated learning and domain generalization.
<img width="1598" alt="image" src="https://github.com/user-attachments/assets/05bff144-3d32-4c0c-8cd5-5fdee13662d4" />

## Paper
Our paper was accepted to the 2025 International Conference on Acoustics, Speech, and Signal Processing (ICASSP).
For more details, refer to our [paper](https://arxiv.org/abs/2412.11408).
### Citation

If you find this work useful, please consider citing:

```bash
@article{soltany2024federated,
  title={Federated Domain Generalization with Label Smoothing and Balanced Decentralized Training},
  author={Soltany, Milad and Pourpanah, Farhad and Molahasani, Mahdiyar and Greenspan, Michael and Etemad, Ali},
  journal={arXiv preprint arXiv:2412.11408},
  year={2024}
}
```

## Code
The code for this repository is **coming soon**. Stay tuned for updates!
To run FedSB, first download the relevant datasets (We recommend that you start with PACS)



### Requirements
Follow these steps to set up your environment and train models:

1. **Install Dependencies**  
   Use the following command to install all required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   We ran our experiments on an Ubuntu 20.04 machine with python==3.10
   
3.	**Download Datasets**
Use the links provided in the Data section. Extract the datasets into the ./data folder (or a directory of your choice).
4.	**Train Models**
To train a fedsb, use the following command:


```bash
for domain in {1..3}; do
    for iteration in {1..3}; do
        echo " No budget no smooth Testing on domain $domain..."
        python main.py --dataset PACS --test_envs $domain --model_save_path "$MODEL_SAVE_PATH" --wandb $WAND_NAME --num_global_epochs 100 --backbone resnet18 --use_mixstyle --ls_eps 0.01 --budget 45  --domain_weights fedavg

    done;
done;

```
Replace the following placeholders with your desired entries:
1. "$MODEL_SAVE_PATH" with the directory in which you would like to save your models.
2. $WAND_NAME with a name for your wandb project





## Data

FedGaLA supports the following datasets:
1. [PACS](https://www.v7labs.com/open-datasets/pacs)
2. [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
3. [TerraINC](https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz)
5. [VLCS](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8)

Please download and extract them to the `./data` directory or a directory you choose, then change the `--dataroot` argument to that directory.







## Credits:
Code for DG datasets is modified from https://github.com/facebookresearch/DomainBed

Code for mixstyle is taken from https://github.com/KaiyangZhou/mixstyle-release/tree/master


### Our team

- [Milad Soltany](https://github.com/miladsoltany) 
- [Mahdiyar Molahasani](https://github.com/MahdiyarMM) 
- [Farhad Pourpanah](https://github.com/Farhad0086) 

Feel free to check out our GitHub profiles for more of our work!

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, features, or bug fixes.

## Contact

For questions or inquiries, please reach out to milad.soltany@queensu.ca.
