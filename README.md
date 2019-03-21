# face_gan
An adversarial translator for CelebA. It's similar to StarGAN except not star but single line :) There is no considerable tech difference with multiple attributes translation, but gender translation is most interesting so I just do it.

Here are some result examples:

![1](https://github.com/seasonyc/face_gan/blob/master/pictures/image_pair0_1553074287.2175086.jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/image_pair0_1553074289.4493124.jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/image_pair0_1553074249.7120426.jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/test711_real_[-23.876125]_cls_[1.8223753e-05]_fake_[-30.68583]_cls_[0.99965394].jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/test384_real_[-24.379768]_cls_[0.99990344]_fake_[-31.342482]_cls_[7.409108e-05].jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/test732_real_[-25.766495]_cls_[0.00017546]_fake_[-30.825687]_cls_[0.99986744].jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/test17382_real_[-25.97427]_cls_[0.00457427]_fake_[-31.820068]_cls_[0.9998902].jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/test18195_real_[-23.27821]_cls_[0.999808]_fake_[-30.654938]_cls_[4.9062754e-05].jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/test18570_real_[-24.10792]_cls_[0.9996203]_fake_[-32.12931]_cls_[0.00791965].jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/test19333_real_[-27.565681]_cls_[0.99935895]_fake_[-30.292011]_cls_[0.00061151].jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/test19411_real_[-24.270924]_cls_[0.00597421]_fake_[-31.70248]_cls_[0.99802905].jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/v20855_real_[-25.562374]_cls_[0.9997322]_fake_[-32.25587]_cls_[0.00554323].jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/v20973_real_[-23.889225]_cls_[0.9988399]_fake_[-30.70314]_cls_[0.000201].jpg)
![1](https://github.com/seasonyc/face_gan/blob/master/pictures/v20680_real_[-28.391115]_cls_[0.9996394]_fake_[-31.644875]_cls_[0.0160531].jpg)

More examples are in pictures directory.

## Usage:
This repo shares the same infrastructure with my other repo https://github.com/seasonyc/face_vae, please refer to it for usage. Run face_adversarial_translator.py to train and test the translator.

face_gan.py was trying generative sampling, but not finished, may be updated after translator is finished completely.
