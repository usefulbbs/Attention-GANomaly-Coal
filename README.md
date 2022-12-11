# Attention-GANomaly
This repository contains PyTorch implementation of the following paper: Foreign object detection of underground coal conveyor belt based on improved GANomaly.

The mining belt conveyor is one of the most important modules in coal mine, whose safety always be threatened by the foreign objects. Although the traditional target detection methods achieve promising results in various computer vision tasks, the performance heavily depends on sufficient labelled data. However, in real-world production scenario, it is difficult to acquire huge number of images with foreign objects. The obtained datasets lacking of capacity and diversity are not suitable for training supervised learning-based foreign objects detection models. To address this concern, we propose a novel method for detecting the foreign objects on the surface of underground coal conveyor belt via improved GANomaly. The proposed foreign objects detection method employs generative adversarial networks (GAN) with attention gate to capture the distribution of normality in both high-dimensional image space and low-dimensional latent vector space. Only the normal images without foreign object are utilized to adversarially train the proposed network, including a U-shape generator to reconstruct the input image and a discriminator to classify real images from reconstructed ones. Then the combination of the difference between the input and generated images as well as the difference between latent representations are utilized as the anomaly score to evaluate whether the input image contain foreign objects. Experimental results over 707 images from real-world industrial scenarios demonstrate that the proposed method achieves an area under the Precision-Recall curve of 0.864 and is superior to the previous GAN-based anomaly detection methods.

1. train
Args:
    dataset='folder'
    --dataroot
        --train
            --0
        --test
            --0
            --1
python ./train.py

2. test
get ./{epoch)_result.json
get auc
python ./cal_pre_recall.py get_single_auc ./{epoch)_result.json
