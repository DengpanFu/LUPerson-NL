# LUPerson-NL
Large-Scale Pre-training for Person Re-identification with Noisy Labels (LUPerson-NL)

The repository is for our CVPR2022 paper [Large-Scale Pre-training for Person Re-identification with Noisy Labels]().

## LUPerson-NL Dataset
LUPerson-NL is currently the largest noisy annotated Person Re-identification dataset without humuan labelling efforts, which is used for Pre-training. LUPerson-NL consists of 10M images of over 430K identities extracted from 21K street-view videos and covers a much diverse range of capturing environments. 

**Details can be found at [./LUP-NL](https://github.com/DengpanFu/LUPerson-NL/tree/main/LUP-NL)**.

## Pre-trained Models
| Model | link |
| :------: | :------: |
| ResNet50 | [R50](https://drive.google.com/file/d/1pFyAdt9BOZCtzaLiE-W3CsX_kgWABKK6/view?usp=sharing) |
| ResNet101 | [R101](https://drive.google.com/file/d/1Ckn0iVtx-IhGQackRECoMR7IVVr4FC5h/view?usp=sharing) |
| ResNet152 | [R152](https://drive.google.com/file/d/1nGGatER6--ZTHdcTryhWEqKRKYU-Mrl_/view?usp=sharing) |

## Finetuned Results
For MGN with ResNet50:

|Dataset | mAP | cmc1 | link |
|:------:|:---:|:----:|:----:|
| MSMT17 | 66.06/79.93 | 85.08/87.63 | - |
| DukeMTMC | 82.27/91.70 | 90.35/92.82 | - |
| Market1501 | 91.12/96.16 | 96.26/97.12 | - |
| CUHK03-L | 74.54/85.84 | 74.64/82.86 | - |

These numbers are a little different from those reported in our paper, and most are slightly better.

For MGN with ResNet101:
|Dataset | mAP | cmc1 | path |
|:------:|:---:|:----:|:----:|
| MSMT17 | 68.41/81.12 | 86.28/88.27 | - |
| DukeMTMC | 84.15/92.77 | 91.88/93.99 | - |
| Market1501 | 91.86/96.21 | 96.56/97.03 | - |
| CUHK03-L | 75.98/86.73 | 75.86/84.07 | - |

**The numbers are in the format of `without RR`/`with RR`**.


## Citation
If you find this code useful for your research, please cite our paper.
```
@article{fu2020unsupervised,
  title={Unsupervised Pre-training for Person Re-identification},
  author={Fu, Dengpan and Chen, Dongdong and Bao, Jianmin and Yang, Hao and Yuan, Lu and Zhang, Lei and Li, Houqiang and Chen, Dong},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
}
```
```
@article{fu2021pnl,
  title={Large-Scale Pre-training for Person Re-identification with Noisy Labels},
  author={Fu, Dengpan and Chen, Dongdong and Yang, Hao and Bao, Jianmin and Yuan, Lu and Zhang, Lei and Li, Houqiang and Wen, Fang and Chen, Dong},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2022}
}
```
