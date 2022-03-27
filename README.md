# LUPerson-NL
Large-Scale Pre-training for Person Re-identification with Noisy Labels (LUPerson-NL)

The repository is for our CVPR2022 paper [Large-Scale Pre-training for Person Re-identification with Noisy Labels]().

## LUPerson-NL Dataset
LUPerson-NL is currently the largest noisy annotated Person Re-identification dataset without humuan labelling efforts, which is used for Pre-training. LUPerson-NL consists of 10M images of over 430K identities extracted from 21K street-view videos and covers a much diverse range of capturing environments. 

**Details can be found at [./LUP-NL](https://github.com/DengpanFu/LUPerson-NL/tree/main/LUP-NL)**.

## Pre-trained Models
| Model | link |
| :------: | :------: |
| ResNet50 | [R50](https://pan.baidu.com/s/1aZxpwlZvekA4V-bQgZOM8w) 提取码: pr50 |
| ResNet101 | [R101](https://pan.baidu.com/s/1wQW7M6IKtEFUKbZVpzSjcg) 提取码: r101 |
| ResNet152 | [R152](https://pan.baidu.com/s/1hAqZlQYRgmnkx8qsZQjKMw) 提取码: r152 |

## Finetuned Results
For MGN with ResNet50:

|Dataset | mAP | cmc1 | link |
|:------:|:---:|:----:|:----:|
| MSMT17 | 68.0 | 86.0 | - |
| DukeMTMC | 84.3 | 92.0 | - |
| Market1501 | 91.9 | 96.6 | - |
| CUHK03-L | 80.4 | 80.9 | - |

<!-- These numbers are a little different from those reported in our paper, and most are slightly better. -->

For MGN with ResNet101:
|Dataset | mAP | cmc1 | path |
|:------:|:---:|:----:|:----:|
| MSMT17 | 70.8 | 87.1 | - |
| DukeMTMC | 85.5 | 92.8 | - |
| Market1501 | 92.5 | 96.9 | - |
| CUHK03-L | 80.5 | 81.2 | - |

For MGN with ResNet152:
|Dataset | mAP | cmc1 | path |
|:------:|:---:|:----:|:----:|
| MSMT17 | 71.6 | 87.5 | - |
| DukeMTMC | 85.6 | 92.4 | - |
| Market1501 | 92.7 | 96.8 | - |
| CUHK03-L | 80.6 | 81.2 | - |

<!-- **The numbers are in the format of `without RR`/`with RR`**. -->


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
