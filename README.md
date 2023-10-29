# multilingual-phonetic-sv

## Installation
1、Created your a Python 3.9 environment.
```bash
conda create -n sre python=3.9
```
2、Pytorch
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
3、Install required packages
```bash
git clone https://github.com/zds-potato/multilingual-phonetic-sv.git
cd multilingual-phonetic-sv
pip -r requirements.txt
```


## Acknowledge
We borrowed a lot of code from:
1. [mfa_conformer](https://github.com/zyzisyz/mfa_conformer) 
2. [sunine](https://gitlab.com/csltstu/sunine/-/tree/master)
3. [wespeaker](https://github.com/wenet-e2e/wespeaker)
4. [wenet](https://github.com/wenet-e2e/wenet)
5. [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)

## Other
We will open-source our trained multilingual speech recognition model soon...