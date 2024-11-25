
## 🔎 Overview framework

Pipelines of different inversion methods in text-driven editing. (a) DDIM inversion inverts a real image to a latent noise code, but the inverted noise code often results in large gap of reconstruction $D_{Rec}$ with higher CFG parameters. (b) NTI optimizes the null-text embedding to narrow the gap of reconstruction $D_{Rec}$, while NPI further optimizes the speed of NTI. (c) DirectInv records the differences between the inversion feature and the reconstruction feature, and merges them back to achieve high-quality reconstruction. (d) Our SPDInv aims to minimize the gap of noise $D_{Noi}$, instead of $D_{Rec}$, which can reduce the impact of source prompt on the editing process and thus reduce the artifacts and inconsistent details encountered by the previous methods.

![SPDInv](figures/methods.png)

## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/leeruibin/SPDInv.git
cd SPDInv

# create an environment with python >= 3.8
conda env create -f environment.yaml
conda activate SPDInv
```

## 🚀 Quick Inference

#### Run P2P with SPDInv

```
python run_SPDInv_P2P.py --input xxx --source [source prompt] --target [target prompt] --blended_word "word1 word2"
```

#### Run MasaCtrl with SPDInv
```
python run_SPDInv_MasaCtrl.py --input xxx --source [source prompt] --target [target prompt]
```

#### Run PNP with SPDInv
To run PNP, you should first upgrade diffusers to 0.17.1 by

```
pip install diffusers==0.17.1
```
then, you can run
```
python run_SPDInv_PNP.py --input xxx --source [source prompt] --target [target prompt]
```

#### Run ELITE with SPDInv
For ELITE, you should first download the pre-trained [global_mapper.pt](https://drive.google.com/drive/folders/1VkiVZzA_i9gbfuzvHaLH2VYh7kOTzE0x?usp=sharing) checkpoint provided by the ELITE, put it into the checkpoints folder.
```
python run_SPDInv_ELITE.py --input xxx --source [source prompt] --target [target prompt]
```

## 📷 Editing cases with P2P, MasaCtrl, PNP, ELITE
## Editing cases with P2P
<div  align="center"> <img src="./figures/cases_P2P.jpg" width = "600" alt="P2P" align=center /> </div>

## Editing cases with MasaCtrl
<div  align="center"> <img src="./figures/cases_MasaCtrl.jpg" width = "600" alt="MasaCtrl" align=center /> </div>

## Editing cases with PNP
<div  align="center"> <img src="./figures/cases_PNP.jpg" width = "600" alt="PNP" align=center /> </div>

## Editing cases with ELITE
<div  align="center"> <img src="./figures/cases_ELITE.jpg" width = "600" alt="ELITE" align=center /> </div>

