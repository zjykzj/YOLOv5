<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

<div align="center"><a title="" href="https://github.com/zjykzj/YOLOv5"><img align="center" src="./imgs/YOLOv5.png" alt=""></a></div>

<p align="center">
  «YOLOv5» implements a tiny version of the original <a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

[ultralytics/yolov5](https://github.com/ultralytics/yolov5) provides a perfect object detection implementation,
including the advanced yolov5 model and loss function, as well as perfect logging and debugging functions. For
beginners, training, testing, and deployment of object detection tasks can be completed through documentation, and
even [ultralytics/yolov5](https://github.com/ultralytics/yolov5) provides SOP for classification and segmentation tasks.

This repository aims to implement a simplified version of YOLOv5, simplifying the internal implementation of the
original YOLOv5 repository as much as possible, and removing features and code that I currently do not need. For
example, I will remove the implementation code for video files and cache files in the data module, and only retain the
onnxruntime/opencv implementation in the deployment module, and so on.

**Note1: The implementation of yolov5 for this warehouse is referenced from [v7.0 - YOLOv5 SOTA Realtime Instance Segmentation](https://github.com/ultralytics/yolov5/releases/tag/v7.0).**

**Note2: The configuration of this warehouse is completely based on the original implementation of YOLOv5, divided into `configs/data/*.yaml`, `configs/hyps/*.yaml`, `configs/models/*.yaml`.**

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [zjykzj/YOLOv3](https://github.com/zjykzj/YOLOv3)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/YOLOv5/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2023 zjykzj