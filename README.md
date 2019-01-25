## Unsupervised Multi-gram losses Super-Resolution method:

This is the code of our method UMSR and we make it by the original code of [SRGAN](https://github.com/tensorlayer/srgan). We greatly thank the author for their sharing code. 
We train the code in the background of [TensorFlow](https://www.tensorflow.org) 1.4 and the [TensorLayer](https://github.com/tensorlayer/tensorlayer) 1.8.0+.


### UMSR Architecture
There are a lot of supervised Deep-learning SR methods and they achieved great success. We pay attention to the special cases of training the network with one image as input.
In fact, there is few works about this and [ZSSR](https://github.com/assafshocher/ZSSR) is the very method which our research build on. Thanks a lot for their contribution.
Another important factor in our paper is the Multi-gram losses from High-resolution multi-scale neural texture synthesis (· Proceeding SA '17 SIGGRAPH Asia 2017 Technical 
Briefs Article No. 13) and thanks for their contributions.

TensorFlow Implementation of ["UMSR: Unsupervised Single Image Super Resolution \\ with Multi-Gram Losses"](will upload soon)

The architecture of our method - UMSR is show as follows:



### Comparing Results

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/SRGAN_Result2.png" width="80%" height="50%"/>
</div>
</a>

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/SRGAN_Result3.png" width="80%" height="50%"/>
</div>
</a>

### Prepare Data and Pre-trained VGG

- 1. You need to download the pretrained VGG19 model in [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) as [tutorial_vgg19.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg19.py) show.
- 2. You need to have the input image for training.
  - You need to generate the training dataset by the data generator first.
  - The weakness of our method is that you need to train a special network for the image and the pretrained model can be used just for this image.
  - You need to change the address of your generation in the config file. 
  - For the test iamge, you need to put the original input image in the LR test folder.



### Run

```python
config.TRAIN.img_path = "your_image_folder/"
```

- Start training.

```bash
python main.py
```


```bash
python main.py --mode=evaluate 
```

### Author
- [Yong shi]
- [Biao Li](https://github.com/Sudo-Biao) 
- [Zhiquan qi](https://github.com/qizhiquan)
- [Yingjie Tian]
- [Jiabin Liu]

### License

- For academic and non-commercial use only.
- For commercial use, please contact qizhiquan@foxmail.com.
