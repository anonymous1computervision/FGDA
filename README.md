# FGDA
For IJCAI 2019 &lt;An Adversarial Domain Adaptation Network for Cross-Domain Fine-Grained Recognition>.

We add extra details of our paper in this repository containing the following aspects:
+ Examples of our proprosed dataset ***MegRetail***,
+ Evaluation of **Self-Attention** module.

## Examples of our proprosed dataset ***MegRetail***

***MegRetail*** contains 52,011 instances from 24,908 images of 263 classes from 3 domains, enabling researcher to further study on domain adaptation fine-grained recognition task.

***MegRetail*** is collected from the retail scenarios, *i.e.,* instant noodles, fruit juice, mineral water, yogurt and milk.



We randomly select 5 instances of 263 classes from 3 domains listing in [/MegRetail_Example](https://github.com/Anonymous2019IJCAI/FGDA/tree/master/MegRetail_Example).

## Evaluation of Self-Attention module

In this part, we use the model trained in semi-supervised setting from $\mathcal{S}$ to $\mathcal{R}$ on ***MegRetail***. And we visualize the features $\mathcal{F}(\mathbf{x})$ we use to select the details of images.

We choose to visualize it in the form of heat map in the figure present below. The first column is the original images. And the middle column presents the mask $\mathcal{M}(\mathcal{F}(\mathbf{x}))$.  The third column is our details $\mathbf{X}_p​$.

We can acquire that **Self-Attention** module can focus on the important parts of images, *e.g.*, the bands, categories and everything that is helpful to recognize the objects. It is obviously that text contains abundant information to help us distinguish different categories. Also in our experiments, employing **Self-Attention** can gain accuracy by 2% in Tab.7.