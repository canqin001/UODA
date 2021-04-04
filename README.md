
# Semi-supervised Domain Adaptation - UODA

This repo contains the source code and dataset for our SDM 2021 paper:
[**Contradictory Structure Learning for Semi-supervised Domain Adaptation**](https://arxiv.org/pdf/2002.02545.pdf)
<br>
SIAM International Conference on Data Mining (SDM), 2021.
<br>
[paper](),
[arXiv](https://arxiv.org/pdf/2002.02545.pdf),
[bibtex]()

![UODA](/Figs/UODA.png)

## Introduction
Current adversarial adaptation methods attempt to align the cross-domain features, whereas two challenges remain unsolved: 1) the conditional distribution mismatch and 2) the bias of the decision boundary towards the source domain. To solve these challenges, we propose a novel framework for semi-supervised domain adaptation by unifying the learning of opposite structures (UODA). UODA consists of a generator and two classifiers (i.e., the source-scattering classifier and the target-clustering classifier), which are trained for contradictory purposes. The target-clustering classifier attempts to cluster the target features to improve intra-class density and enlarge inter-class divergence. Meanwhile, the source-scattering classifier is designed to scatter the source features to enhance the decision boundary's smoothness. Through the alternation of source-feature expansion and target-feature clustering procedures, the target features are well-enclosed within the dilated boundary of the corresponding source features. This strategy can make the cross-domain features to be precisely aligned against the source bias simultaneously. Moreover, to overcome the model collapse through training, we progressively update the measurement of feature's distance and their representation via an adversarial training paradigm. Extensive experiments on the benchmarks of DomainNet and Office-home datasets demonstrate the superiority of our approach over the state-of-the-art methods.


## Dataset
This part follows the same protocol of [MME](https://github.com/VisionLearningGroup/SSDA_MME).

To get data, run

`sh download_data.sh`

The images will be stored in the following way.

`./data/multi/real/category_name`,

`./data/multi/sketch/category_name`

The dataset split files are stored as follows,

`./data/txt/multi/labeled_source_images_real.txt`,

`./data/txt/multi/unlabeled_target_images_sketch_3.txt`,

`./data/txt/multi/validation_target_images_sketch_3.txt`.

The office and office home datasets are organized in the following ways,

 `./data/office/amazon/category_name`,
 
 `./data/office_home/Real/category_name`,


## Requirements
`pip install -r requirements.txt`

## Train & Test
If you run the experiment on one adaptation scanerio, like real to clipart of the DomainNet,
```
python main.py --dataset multi --source real --target sketch
```

## Citation
If you find this project useful for your research, please kindly cite our paper:

```bibtex
@misc{qin2021contradictory,
      title={Contradictory Structure Learning for Semi-supervised Domain Adaptation}, 
      author={Can Qin and Lichen Wang and Qianqian Ma and Yu Yin and Huan Wang and Yun Fu},
      year={2021},
      eprint={2002.02545},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
Our implementation of UODA heavily relies on [MME](https://github.com/VisionLearningGroup/SSDA_MME).






