# FGDA

For IJCAI 2019 &lt;An Adversarial Domain Adaptation Network for Cross-Domain Fine-Grained Recognition>.

We add extra details of our paper in this repository containing the following aspects:
+ Examples of our proprosed dataset ***MegRetail***,
+ Evaluation of **Self-Attention** module.

The whole architechture of our model is presented below.

![model]https://github.com/Anonymous2019IJCAI/FGDA/blob/master/pics/model.png?raw=true)

## Examples of our proprosed dataset ***MegRetail***

***MegRetail*** contains 52,011 instances from 24,908 images of 263 classes from 3 domains, enabling researcher to further study on domain adaptation fine-grained recognition task.

***MegRetail*** is collected from the retail scenarios, *i.e.,* instant noodles, fruit juice, mineral water, yogurt and milk.

![megretail_examples](https://github.com/Anonymous2019IJCAI/FGDA/blob/master/pics/our_examples.png?raw=true)

We randomly select 5 instances of 263 classes from 3 domains listing in [./MegRetail_Example](https://github.com/Anonymous2019IJCAI/FGDA/tree/master/MegRetail_Example).

## Evaluation of Self-Attention module

In this part, we use the model trained in semi-supervised setting from ![eq0](https://github.com/Anonymous2019IJCAI/FGDA/blob/master/pics/eq0.png?raw=true) to ![eq1](https://github.com/Anonymous2019IJCAI/FGDA/blob/master/pics/eq1.png?raw=true) on ***MegRetail***. And we visualize the features ![eq2](https://github.com/Anonymous2019IJCAI/FGDA/blob/master/pics/eq2.png?raw=true) we use to select the details of images.

We choose to visualize it in the form of heat map in the figure present below. The first column is the original images. And the middle column presents the mask ![eq3](https://github.com/Anonymous2019IJCAI/FGDA/blob/master/pics/eq3.png?raw=true).  The third column is our details ![eq4](https://github.com/Anonymous2019IJCAI/FGDA/blob/master/pics/eq4.png?raw=true).

![sa_examples](https://github.com/Anonymous2019IJCAI/FGDA/blob/master/pics/sa_examples.png?raw=true)

We can acquire that **Self-Attention** module can focus on the important parts of images, *e.g.*, the bands, categories and everything that is helpful to recognize the objects. It is obviously that text contains abundant information to help us distinguish different categories. Also in our experiments, employing **Self-Attention** can gain accuracy by 2% in Tab.7.

Our model can be conducted using the code in [./model_code](https://github.com/Anonymous2019IJCAI/FGDA/tree/master/model_code), while we save the model in  [./sa_model](https://github.com/Anonymous2019IJCAI/FGDA/tree/master/sa_model).
