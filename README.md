# A Neural Algorithm of Artistic Style

Implementing: Gatys, L.A., Ecker, A.S. and Bethge, M., 2016, June. [Image style transfer using convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). In Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on (pp. 2414-2423). IEEE.

This project provides notebooks for running through different sections of the paper and scripts that can be used to 'style' images as described by the technique in the paper. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

This project was built on Python 3.5.2.

The following packages should be installed:
* ```matplotlib (1.5.3)```
* ```numpy (1.14.0)```
* ```opencv-python (3.4.2.16)```
* ```Pillow (3.3.1)```
* ```tensorboard (1.8.0) (optional)```
* ```tensorflow-gpu (1.8.0)```

The following packages which are part of the Python standard library are used in this project:
* ```os```
* ```shutil```

Additionally, I utilise a pretrained VGG19 model checkpoint. You can download the compressed folder from [here](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz). After extraction, place the vgg_19.ckpt file in this folder or within the scripts you are using, change the variable ```checkpoint_path``` to point to the relevant location. 

### Installing
You may install these packages yourself or use the [requirements.txt](requirements.txt) file like so: 
```
pip install -r requirements.txt
```


## Usage

### Notebooks
#### Content Representation
You can interact with the [notebook](content_recs.ipynb) which works through the ideas presented in Section 2.1: Content Representation of the paper. You can also read this [post](https://ashwindcruz.github.io/blog/2018/07/30/content-reconstruction)  "Content Reconstruction") to better understand the notebook. 
Before using this notebook, download an image you want to experiment with and change the variable ```real_image_path``` in the notebook to point to your chosen image.

#### Style Representation
You can interact with the [notebook](style_recs.ipynb) which works through the ideas presented in Section 2.2: Style Representation of the paper. You can also read this [post](https://ashwindcruz.github.io/blog/2018/09/08/style-reconstruction)  "Style Reconstruction") to better understand the notebook. 
Before using this notebook, download an image you want to experiment with and change the variable ```real_image_path``` in the notebook to point to your chosen image.

#### Style Transfer
You can interact with the [notebook](style_transfer.ipynb) which works through the ideas presented in Section 2.3: Style Transfer of the paper. You can also read this [post](https://ashwindcruz.github.io/blog/2018/09/26/style-transfer)  "Style Transfer") to better understand the notebook. 
Before using this notebook, download a content and style image you want to experiment with and change the variables ```content_image_path``` and ```style_image_path``` respectively in the notebook to point to your images.

### Scripts
If you prefer, you can run Python scripts directly. 
Ensure you have downloaded a content image and a style image and set the paths appropriately in [config.py](config.py).
Other parameters can also be tweaked in [config.py](config.py). 

Once done, please run: 

```python train.py```
<!---## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.
--->
## Versioning

I use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/ashwindcruz/style-transfer/tags). 

## Authors

* **Ashwin D'Cruz** - [ashwindcruz](https://github.com/ashwindcruz)

<!---See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.--->

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
<!---
## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
--->
