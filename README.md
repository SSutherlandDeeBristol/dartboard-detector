# Dartboard Detector
Summative coursework by @Michael-JB and @SSutherlandDeeBristol

This project implements Hough transforms from scratch for both line and circle extraction, then combines [OpenCV](https://github.com/opencv/opencv) viola jones data to create a weighted dartboard detector. The implementation is summarised in `report.pdf`.

## Sample Detections

These are some examples of the final detector output given the inputs from `images/`. All detections are shown in `output/detections/`, with intermediate outputs saved in `output/intermediate/`.

![Dart 0](/output/dart0overlay.jpg)
![Dart 7](/output/dart7overlay.jpg)
![Dart 8](/output/dart8overlay.jpg)
![Dart 13](/output/dart13overlay.jpg)
![Dart 14](/output/dart14overlay.jpg)
![Dart 15](/output/dart15overlay.jpg)

## License

[MIT license](./LICENSE)
