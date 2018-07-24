# A Neural Algorithm of Artistic Style

Implementing: Gatys, L.A., Ecker, A.S. and Bethge, M., 2016, June. [Image style transfer using convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). In Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on (pp. 2414-2423). IEEE.

This project provides notebooks for running through different sections of the paper and scripts that can be used to 'style' images as described by the technique in the paper. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

This project was built on Python 3.5.2.

The following packages must be installed:
* ```cv2 (3.4.1)```
* ```matplotlib (1.5.3)```
* ```numpy (1.14.0)```
* ```opencv-python (3.4.1+contrib)```
* ```tensorflow-gpu (1.8.0)```

The following packages which are part of the Python standard library are used in this project:
* ```os```
* ```shutil```

Additionally, I utilise a pretrained VGG19 model checkpoint. You can download the compressed folder from [here](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz). Place the vgg_19.ckpt file in this folder or if you wish to place it elsewhere, ensure that the relevant paths are changed in the code.  

### Installing
You may install these packages yourself or use the [requirements.txt](requirements.txt) file like so: 
```
pip install -r requirements.txt
```


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
