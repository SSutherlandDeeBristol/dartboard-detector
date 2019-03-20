/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - detector.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <string>

using namespace std;
using namespace cv;

/** Global variables */
String cascade_name = "../dartclassifiertraining/dart1000cascade/dart1000cascade.xml";
CascadeClassifier cascade;

/** Threshold constants */
const int minDartboardRadius = 20; // px
const int maxDartboardRadius = 120; // px
const int houghSupressionWidth = 45;
const int magnitudeThreshold = 50;

const int houghLinesMagnitudeThreshold = 100;
const float circleClassifyFactor = 0.8;

const int houghLinesSupressionWidth = 30;
const float lineClassifyFactor = 0.6;

const int intersectionSupressionWidth = 30;
const int minimumIntersectingLines = 3;
const float intersectClassifyFactor = 0.95;

/** Function Headers */
vector<tuple<Point, Point>> getHoughLines(Mat houghLineSpace, Mat gradientMagnitude, Mat frame, float houghThreshold, float houghSupressionWidth);
Mat getLineIntersectionSpace(Mat image, vector<tuple <Point, Point>> allLines);
vector<tuple<bool,Rect>> getViolaJones(Mat image);
vector<tuple<int, Point>> getHoughCircles(Mat houghSpace, Mat frame, int minRadius, int maxRaduis);
vector<Point> getLineIntersections(Mat frame, Mat lineIntersectionSpace);
int incrementLimit(int current, int incrementBy, int limit);
void hough(cv::Mat &output, cv::Mat &gradient, cv::Mat &direction, int maxRadius, int minRadius);
void houghlines(cv::Mat &output, cv::Mat &gradient, cv::Mat &direction);
void getGradientDirection(cv::Mat &Gx, cv::Mat &Gy, cv::Mat &output);
void convolution(cv::Mat &image, cv::Mat &kernel, cv::Mat &output);
void drawHoughDetection(Mat frame, vector<tuple<bool,Rect>> violaJonesDataComp, vector<tuple<int, Point>> circles, vector<Point> intersections, int minDartboardRadius, int maxDartboardRadius);

/** @function main */
int main(int argc, const char** argv) {
    
    // Read image in
    string inputfilename = argv[1];
    Mat frame = imread(inputfilename, CV_LOAD_IMAGE_COLOR);
    
    // Load the Viola Jones Strong Classifier
    if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); };
    
    // Convert image to greyscale and apply small gaussian blur
    Mat greyImage;
    cvtColor(frame, greyImage, CV_BGR2GRAY);
    GaussianBlur(greyImage, greyImage, Size(11, 11), 0, 0, BORDER_DEFAULT);
    
    // Initialise sobel kernels
    cv::Mat Gx = (cv::Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
    cv::Mat Gy = (cv::Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    
    // Initialise gradient matrices
    cv::Mat gradientX, gradientXAbs, gradientY, gradientYAbs, gradientDirection, gradientMagnitude;
    cv::Mat houghCircleSpace, houghLineSpace, gradientMagnitudeLine;
    gradientX.create(greyImage.size(), CV_32F);
    gradientXAbs.create(greyImage.size(), CV_32F);
    gradientY.create(greyImage.size(), CV_32F);
    gradientYAbs.create(greyImage.size(), CV_32F);
    gradientMagnitude.create(greyImage.size(), CV_32F);
    gradientDirection.create(greyImage.size(), CV_32F);
    gradientMagnitudeLine.create(greyImage.size(), CV_32F);
    
    // Perform convolution and scaling
    convolution(greyImage, Gx, gradientX);
    convolution(greyImage, Gy, gradientY);
    convertScaleAbs(gradientX, gradientXAbs);
    convertScaleAbs(gradientY, gradientYAbs);
    
    // Create gradient magnitude and direction
    addWeighted(gradientXAbs, 0.5, gradientYAbs, 0.5, 0, gradientMagnitude);
    threshold(gradientMagnitude, gradientMagnitude, magnitudeThreshold, 255, THRESH_BINARY);
    addWeighted(gradientXAbs, 0.5, gradientYAbs, 0.5, 0, gradientMagnitudeLine);
    threshold(gradientMagnitudeLine, gradientMagnitudeLine, houghLinesMagnitudeThreshold, 255, THRESH_BINARY);
    getGradientDirection(gradientX, gradientY, gradientDirection);

    // Create circle hough space and detect circles
    hough(houghCircleSpace, gradientMagnitude, gradientDirection, minDartboardRadius, maxDartboardRadius);
    vector<tuple<int, Point>> circles = getHoughCircles(houghCircleSpace,frame, minDartboardRadius, maxDartboardRadius);
    
    // Create line hough space
    houghlines(houghLineSpace, gradientMagnitudeLine, gradientDirection);

    // Detect lines from line hough space
    vector<tuple <Point, Point>> allLines = getHoughLines(houghLineSpace, gradientMagnitudeLine, frame, lineClassifyFactor, houghLinesSupressionWidth);

    // Detect line intersections
    Mat lineIntersectionSpace = getLineIntersectionSpace(frame, allLines);
    vector<Point> intersections = getLineIntersections(frame, lineIntersectionSpace);
    
    // Get Viola Jones detections
    vector<tuple<bool,Rect>> violaJonesDataComp = getViolaJones(frame);
    
    // Call detector with gathered data
    drawHoughDetection(frame, violaJonesDataComp, circles, intersections, minDartboardRadius, maxDartboardRadius);
    
    // Save resulting images
    string outputfile = "detected.jpg";
    imwrite(outputfile, frame);
    cout << "Processed image " << inputfilename << ", saved to: " << outputfile << endl;
    
    frame = imread(inputfilename, CV_LOAD_IMAGE_COLOR);
    
    // Draw lines
    for (int l = 0; l < allLines.size(); l++) {
        line(frame, std::get<0>(allLines.at(l)), std::get<1>(allLines.at(l)), Scalar(0, 0, 255), 1);
    }
    
    // Draw intersections
    for (int i = 0; i < intersections.size(); i++) {
        circle(frame, intersections.at(i), 5, Scalar(50, 150, 255), 2);
    }
    
    // Draw circles
    for (int c = 0; c < circles.size(); c++) {
        circle(frame, std::get<1>(circles.at(c)), std::get<0>(circles.at(c)), Scalar(255, 0, 0), 2);
    }
    
    // Draw Viola Jones
    for (int v = 0; v < violaJonesDataComp.size(); v++) {
        Rect vj =  std::get<1>(violaJonesDataComp.at(v));
        rectangle(frame, Point(vj.x, vj.y), Point(vj.x + vj.width, vj.y + vj.height), Scalar(255, 255, 0), 2);
    }
    
    // Save overlay output
    outputfile = "detected_overlay.jpg";
    imwrite(outputfile, frame);
    cout << "Saved overlay to: " << outputfile << endl;
    
	return 0;
}

// Returns a list of all Viola Jones detections in the form <isROI, bounds>
vector<tuple<bool,Rect>> getViolaJones(Mat image) {
    std::vector<Rect> darts;
    Mat frame_grey;
    vector<tuple<bool,Rect>> allDarts;
    
    // Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor( image, frame_grey, CV_BGR2GRAY );
    equalizeHist( frame_grey, frame_grey );
    
    // Perform Viola-Jones Object Detection
    cascade.detectMultiScale( frame_grey, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
    
    for (int i = 0; i < darts.size(); i++) {
        Point dart_tl = Point(darts[i].x, darts[i].y),
        dart_br = Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height);
        
        Scalar meanWholeImageValue = mean(frame_grey);
        Mat roi(frame_grey, Rect(dart_tl, dart_br));
        Scalar meanROIImageValue = mean(roi);
        Mat greyROI;
        
        double thresholdVal = (meanROIImageValue[0] + meanWholeImageValue[0]) / 2;
        threshold(roi, greyROI, meanROIImageValue[0], 255, THRESH_BINARY);
        
        float whiteCount = 0, blackCount = 0;
        
        for (int y = 0; y < greyROI.rows; y++) {
            for(int x = 0; x < greyROI.cols; x++) {
                if (greyROI.at<uchar>(y,x) == 0) blackCount++;
                else whiteCount++;
            }
        }
        
        double ratio = whiteCount / blackCount;

        if (blackCount > 0 && ratio > 0.3 && ratio < 1.7) {
            tuple <bool, Rect> dart(true, darts[i]);
            allDarts.push_back(dart);
        } else {
            tuple <bool, Rect> dart(false, darts[i]);
            allDarts.push_back(dart);
        }
    }
    
    return allDarts;
}

// Interprets the line intersection space and returns a list of all potential dartboard centres
vector<Point> getLineIntersections(Mat frame, Mat lineIntersectionSpace) {
    vector<Point> intersections;
    double minVal, maxVal, initMax;
    int minIdx[2], maxIdx[2];
    
    minMaxIdx(lineIntersectionSpace, &minVal, &initMax, minIdx, maxIdx);
    maxVal = initMax;
    
    if (maxVal >= minimumIntersectingLines) {
        while (maxVal > (intersectClassifyFactor * initMax)) {
            int maxy = maxIdx[0], maxx = maxIdx[1];
            
            // Add intersection
            Point intersection = Point(maxx, maxy);
            intersections.push_back(intersection);
            
            // Suppress area
            for (int y = maxy - intersectionSupressionWidth / 2; y < maxy + intersectionSupressionWidth / 2; y++) {
                for (int x = maxx - intersectionSupressionWidth / 2; x < maxx + intersectionSupressionWidth / 2; x++) {
                    if (y >= 0 && x >= 0 && x < lineIntersectionSpace.cols && y < lineIntersectionSpace.rows) {
                        lineIntersectionSpace.at<float>(y, x) = 0;
                    }
                }
            }
            
            minMaxIdx(lineIntersectionSpace, &minVal, &maxVal, minIdx, maxIdx);
        }
    }
    
    return intersections;
}

// Returns the line intersection space
Mat getLineIntersectionSpace(Mat image, vector<tuple <Point, Point>> allLines) {
	double minVal, maxVal, initMax;
	int minIdx[2], maxIdx[2];
	cv::Mat intersectionAccumulator;
	intersectionAccumulator.create(image.size(), CV_32F);
	intersectionAccumulator = Scalar(0);

	float gradientDelta = 20;

	for (int x = 0; x < image.cols; x++) {
		for (int y = 0; y < image.rows; y++) {

			vector<tuple <Point, Point>> intersectingLines;
			float maxAngle = -FLT_MAX, minAngle = FLT_MAX;

			for (int l = 0; l < allLines.size(); l++) {
				tuple <Point, Point> line = allLines.at(l);
				Point current = Point(x, y), a = get<0>(line), b = get<1>(line);
				double atoc = norm(current - a);
				double btoc = norm(current - b);
				double atob = norm(b - a);

				if ((atoc + btoc) <= atob + 0.01f) intersectingLines.push_back(line);
			}

			for (int l = 0; l < intersectingLines.size(); l++) {
				tuple <Point, Point> line = intersectingLines.at(l);
				Point a = get<0>(line), b = get<1>(line);

				int run = max(b.x, a.x) - min(b.x, a.x);
				int rise = max(b.y, a.y) - min(b.y, a.y);

				double angle;

				if (run == 0) angle = 90;
				else if (rise == 0) angle = 0;
                else angle = atan2(rise, run) * (180 / M_PI);

				if (angle < minAngle) minAngle = angle;
				if (angle > maxAngle) maxAngle = angle;
			}

			if (maxAngle - minAngle > gradientDelta) intersectionAccumulator.at<float>(y, x) += intersectingLines.size();
		}
	}
    
	return intersectionAccumulator;
}

// Gets a list of all detected hough circles in the form <radius, centre>
vector<tuple<int, Point>> getHoughCircles(Mat houghSpace, Mat frame, int minRadius, int maxRadius) {
    vector<tuple<int, Point>> circles;
    double minVal, maxVal, initMax;
    int minIdx[3], maxIdx[3];
    
    minMaxIdx(houghSpace, &minVal, &initMax, minIdx, maxIdx);
    maxVal = initMax;
    
    while (maxVal > (circleClassifyFactor * initMax)) {
        minMaxIdx(houghSpace, &minVal, &maxVal, minIdx, maxIdx);
        
        int maxy = maxIdx[0], maxx = maxIdx[1], maxr = maxIdx[2];
        
        tuple <int, Point> circle(minRadius + maxr, Point(maxx, maxy));
        circles.push_back(circle);
        
        // Suppress 3D area around maximum
        for (int y = maxy - houghSupressionWidth / 2; y < maxy + houghSupressionWidth / 2; y++) {
            for (int x = maxx - houghSupressionWidth / 2; x < maxx + houghSupressionWidth / 2; x++) {
                for (int r = maxr - houghSupressionWidth / 2; r < maxr + houghSupressionWidth / 2; r++) {
                    if (y >= 0 && x >= 0 && r >= 0 && y < frame.rows && x < frame.cols && r < (maxRadius - minRadius)) {
                        houghSpace.at<float>(y, x, r) = 0;
                    }
                }
            }
        }
    }
    
    return circles;
}

// Returns a list containing all detected hough lines in the form <Top left point, Bottom right point>
vector<tuple <Point, Point>> getHoughLines(Mat houghLineSpace, Mat gradientMagnitude, Mat frame, float houghThreshold, float houghSupressionWidth) {
	vector<tuple <Point, Point>> allLines;
	double minVal, maxVal, initMax;
	int minIdx[2], maxIdx[2];

	minMaxIdx(houghLineSpace, &minVal, &initMax, minIdx, maxIdx);
	maxVal = initMax;

	int maxPossibleP = (gradientMagnitude.rows * (cos(M_PI / 4))) + (gradientMagnitude.cols * (sin(M_PI / 4))) + 3;

	while (maxVal > (houghThreshold * initMax)) {
		int maxp = maxIdx[0] - maxPossibleP; // Normalise P values by subtracting maximum
		int maxt = maxIdx[1];

		float x0 = maxp * cos((maxt * M_PI) / 180); // X on line closest to origin
		float y0 = maxp * sin((maxt * M_PI) / 180); // Y on line closest to origin

		float pgradient = y0 / x0;
		float tgradient = (-1) / pgradient;

		Point pointA, pointB;

		if (pgradient > 100 || pgradient < -100) { // Perfectly horizontal line
			pointA = Point(0, y0);
			pointB = Point(frame.cols, y0);
		} else if (tgradient > 100 || tgradient < -100) { // Perfectly vertical line
			pointA = Point(x0, 0);
			pointB = Point(x0, frame.rows);
		} else { // Any other angle line
			if (tgradient > 0) {
				pointA = Point(0, maxp / (sin((maxt * M_PI) / 180)));
				pointB = Point(frame.cols, (maxp - (frame.cols * cos((maxt * M_PI) / 180))) / (sin((maxt * M_PI) / 180)));
			} else {
				pointA = Point(0, maxp / (sin((maxt * M_PI) / 180)));
				pointB = Point(maxp / (cos((maxt * M_PI) / 180)), 0);
			}
		}

		tuple <Point, Point> line(pointA, pointB);
		allLines.push_back(line);
        
		// Suppress area
		for (int y = maxt - houghSupressionWidth / 2; y < maxt + houghSupressionWidth / 2; y++) {
			for (int x = maxIdx[0] - houghSupressionWidth / 2; x < maxIdx[0] + houghSupressionWidth / 2; x++) {
				if (y >= 0 && x >= 0 && x < houghLineSpace.rows && y < houghLineSpace.cols) {
					houghLineSpace.at<float>(x, y) = 0;
				}
			}
		}

		minMaxIdx(houghLineSpace, &minVal, &maxVal, minIdx, maxIdx);
	}
	return allLines;
}

// Calculates the circular hough space and saves the result to the output argument
void hough(cv::Mat &output, cv::Mat &magnitude, cv::Mat &direction, int minRadius, int maxRadius) {
	int sizes[] = { magnitude.rows, magnitude.cols, maxRadius - minRadius };
	output.create(3, sizes, CV_32F);
	output = Scalar(0);

	for (int y = 0; y < magnitude.rows; y++) {
		for (int x = 0; x < magnitude.cols; x++) {
			if (magnitude.at<uchar>(y, x) == 255) {
				for (int r = 0; r < (maxRadius - minRadius); r++) {
					int radius = r + minRadius;

					float th = direction.at<float>(y, x);
					float delta = M_PI / 32;

					for (float dth = th - delta; dth <= th + delta; dth += M_PI / 64) {
						for (int sign = -1; sign <= 1; sign += 2) {
							int x0 = (x - (sign) * (radius * (cos(dth))));
							int y0 = (y - (sign) * (radius * (sin(dth))));

							if (x0 >= 0 && y0 >= 0 && x0 < magnitude.cols && y0 < magnitude.rows) {
								output.at<float>(y0, x0, r) += 1;
							}
						}
					}
				}
			}
		}
	}
}

// Calculates the hough line space and saves the result to the output argument
void houghlines(cv::Mat &output, cv::Mat &magnitude, cv::Mat &direction) {
	int maxP = (magnitude.cols * (cos(M_PI / 4))) + (magnitude.rows * (sin(M_PI / 4))) + 3;
	int deltaTheta = 180;
	int sizes[] = { 2 * maxP, deltaTheta };
	output.create(2, sizes, CV_32F);
	output = Scalar(0);

	for (int y = 0; y < magnitude.rows; y++) {
		for (int x = 0; x < magnitude.cols; x++) {
			if (magnitude.at<uchar>(y, x) == 255) {
				for (float dth = 0; dth < deltaTheta; dth++) {
					float dthrad = (dth * M_PI) / 180;
					float p = (x * (cos(dthrad))) + (y * (sin(dthrad)));

					// Increment accumulator, shifting by maximum p to account for negative p values
					output.at<float>(p + maxP, dth) += 1;
				}
			}
		}
	}
}

// Calculates the gradient direction and saves the result to the output argument
void getGradientDirection(cv::Mat &Gx, cv::Mat &Gy, cv::Mat &output) {
	for (int i = 0; i < output.rows; i++) {
		for (int j = 0; j < output.cols; j++) {
			output.at<float>(i, j) = atan2(Gy.at<float>(i, j), Gx.at<float>(i, j));
		}
	}
}

// Applies convolution to the input matrix
void convolution(cv::Mat &input, cv::Mat &kernel, cv::Mat &output) {
	int kernelRadiusX = (kernel.size[0] - 1) / 2;
	int kernelRadiusY = (kernel.size[1] - 1) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder(input, paddedInput, kernelRadiusX, kernelRadiusX,
		kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE);

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			double sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {
					int imageX = i + m + kernelRadiusX;
					int imageY = j + n + kernelRadiusY;
					int kernelX = m + kernelRadiusX;
					int kernelY = n + kernelRadiusY;

					float imageVal = (float)paddedInput.at<uchar>(imageX, imageY);
					float kernelVal = kernel.at<float>(kernelX, kernelY);

					sum += imageVal * kernelVal;
				}
			}
			output.at<float>(i, j) = (float)sum;
		}
	}
}

// Our detector which combines input feature data to locate dartboards
void drawHoughDetection(Mat frame, vector<tuple<bool,Rect>> violaJonesDataComp, vector<tuple<int, Point>> circles, vector<Point> intersections, int minDartboardRadius, int maxDartboardRadius) {
    int maxDartboardWidth = 2 * maxDartboardRadius;
    int minDartboardWidth = 2 * minDartboardRadius;
    
    cout << "Creating feature accumulator... " << flush;
    
    // Create feature accumulator
    Mat featureAccumulator;
    int sizes[] = { frame.rows, frame.cols, maxDartboardWidth, maxDartboardWidth }; // 4D Matrix: {x, y, width, height}
    featureAccumulator.create(4, sizes, CV_8U);
    featureAccumulator = Scalar(0);
    vector<Rect> violaJonesData;
    for (int i = 0; i < violaJonesDataComp.size(); i++) violaJonesData.push_back(std::get<1>(violaJonesDataComp.at(i)));
    
    cout << "Done" << endl << flush << "Considering hough line intersection data... " << flush;
    
    // Add intersection data to feature accumulator
    for (int i = 0; i < intersections.size(); i++) {
        Point intersection = intersections.at(i);
        int x = intersection.x, y = intersection.y;
        int incrementRadius = 30;

        for (int xR = x - incrementRadius/2; xR < x + incrementRadius/2; xR++) {
            for (int yR = y - incrementRadius/2; yR < y + incrementRadius/2; yR++) {
                if (xR > 0 && yR > 0 && xR < frame.cols && yR < frame.rows) {
                    for (int w = 0; w < maxDartboardWidth; w++) {
                        for (int h = 0; h < maxDartboardWidth; h++) {
                            const int loc[4] = {yR, xR, w, h};
                            if (xR == x && yR == y) {
                                featureAccumulator.at<uchar>(loc) = incrementLimit(featureAccumulator.at<uchar>(loc), 30, 255);
                            } else {
                                featureAccumulator.at<uchar>(loc) = incrementLimit(featureAccumulator.at<uchar>(loc), 20, 255);
                            }
                        }
                    }
                }
            }
        }
    }
    
    cout << "Done" << endl << flush << "Considering hough circle data... " << flush;
    
    // Add circle data to feature accumulator
    for (int c = 0; c < circles.size(); c++) {
        tuple<int, Point> circ = circles.at(c);
        int x = get<1>(circ).x, y = get<1>(circ).y, r = std::get<0>(circ);
        int incrementRadius = 50; // + 10
        
        for (int xR = x - incrementRadius/2; xR < x + incrementRadius/2; xR++) {
            for (int yR = y - incrementRadius/2; yR < y + incrementRadius/2; yR++) {
                for (int rR = r - incrementRadius/2; rR < r + incrementRadius/2; rR++) {
                    if (xR > 0 && yR > 0 && rR > 0 && rR < maxDartboardRadius && xR < frame.cols && yR < frame.rows) {
                        const int loc[4] = {yR, xR, 2*rR, 2*rR};
                        if (xR == x && yR == y && rR == r) {
                            featureAccumulator.at<uchar>(loc) = incrementLimit(featureAccumulator.at<uchar>(loc), (r/10) + 25, 255);
                        } else {
                            featureAccumulator.at<uchar>(loc) = incrementLimit(featureAccumulator.at<uchar>(loc), (r/10) + 10, 255); // -10
                        }
                    }
                }
            }
        }
    }
    
    cout << "Done" << endl << flush << "Considering Viola Jones data... " << flush;
    
    // Add Viola Jones data to feature accumulator
    for (int v = 0; v < violaJonesData.size(); v++) {
        Rect dart = violaJonesData.at(v);
        Point dart_tl = Point(dart.x, dart.y);
        Point dart_br = Point(dart.x + dart.width, dart.y + dart.height);
        Point dart_centre = Point(dart.x + dart.width/2, dart.y + dart.height/2);
        
        bool isROI = std::get<0>(violaJonesDataComp.at(v));

        int x = dart_centre.x, y = dart_centre.y, w = dart.width, h = dart.height;
        int incrementRadius = 30;

        for (int xR = x - incrementRadius/2; xR < x + incrementRadius/2; xR++) {
            for (int yR = y - incrementRadius/2; yR < y + incrementRadius/2; yR++) {
                for (int wR = w - incrementRadius/2; wR < w + incrementRadius/2; wR++) {
                    for (int hR = h - incrementRadius/2; hR < h + incrementRadius/2; hR++) {
                        if (xR > 0 && yR > 0 && xR < frame.cols && yR < frame.rows) {
                            const int loc[4] = {yR, xR, wR, hR};
                            if (xR == x && yR == y && wR == w && hR == h) {
                                featureAccumulator.at<uchar>(loc) = incrementLimit(featureAccumulator.at<uchar>(loc), isROI ? 35 : 30, 255);
                            } else {
                                featureAccumulator.at<uchar>(loc) = incrementLimit(featureAccumulator.at<uchar>(loc), isROI ? 25 : 20, 255); // -5
                            }
                        }
                    }
                }
            }
        }
    }
    
    cout << "Done" << endl << flush << "Considering composite hough circle & Viola Jones data... " << flush;
    
    // Add circle x Viola Jones data to feature accumulator
    for (int v = 0; v < violaJonesData.size(); v++) {
        for (int c = 0; c < circles.size(); c++) {
            Rect vj = violaJonesData.at(v);
            tuple<int, Point> circ = circles.at(c);
            Point vj_tl = Point(vj.x, vj.y);
            Point vj_br = Point(vj.x + vj.width, vj.y + vj.height);
            Point vj_centre = Point(vj.x + vj.width/2, vj.y + vj.height/2);
            
            Point circ_centre = get<1>(circ);
            int x = circ_centre.x, y = circ_centre.y, r = get<0>(circ);
            
            bool isROI = std::get<0>(violaJonesDataComp.at(v));
            
            int incrementRadius = 50;

            if (norm(vj_centre - circ_centre) < 15) {
                for (int xR = x - incrementRadius/2; xR < x + incrementRadius/2; xR++) {
                    for (int yR = y - incrementRadius/2; yR < y + incrementRadius/2; yR++) {
                        for (int rR = r - incrementRadius/2; rR < r + incrementRadius/2; rR++) {
                            if (xR > 0 && yR > 0 && rR > 0 && rR < maxDartboardRadius && xR < frame.cols && yR < frame.rows) {
                                const int loc[4] = {yR, xR, 2*rR, 2*rR};
                                if (xR == x && yR == y && rR == r) {
                                    featureAccumulator.at<uchar>(loc) = incrementLimit(featureAccumulator.at<uchar>(loc), (r/10) + 15 + (isROI ? 5 : 0), 255);
                                } else {
                                    featureAccumulator.at<uchar>(loc) = incrementLimit(featureAccumulator.at<uchar>(loc), (r/10) + 5 + (isROI ? 2 : 0), 255);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    cout << "Done" << endl << flush << "Inferring dartboards from feature accumulator... " << endl << flush;
    
    // Interpret featureAccumulator
    double minVal, maxVal, initMax;
    int minIdx[4], maxIdx[4];
    int supression = 100;
    float classifyFactor = 0.9f;
    minMaxIdx(featureAccumulator, &minVal, &initMax, minIdx, maxIdx);
    maxVal = initMax;
    int i = 0;

    while (maxVal > (classifyFactor * initMax)) {
        int maxy = maxIdx[0], maxx = maxIdx[1], maxw = maxIdx[2], maxh = maxIdx[3];
        
        Point rect_tl = Point(maxx - (maxw/2), maxy - (maxh/2));
        Point rect_br = Point(maxx + (maxw/2), maxy + (maxh/2));
        
        if (maxw > minDartboardWidth && maxh > minDartboardWidth) {
            rectangle(frame, rect_tl, rect_br, Scalar(255,255,255), 2);
        }
        
        // Suppress 4D area around maximum
        for (int y = maxy - supression / 2; y < maxy + supression / 2; y++) {
            for (int x = maxx - supression / 2; x < maxx + supression / 2; x++) {
                for (int w = 0; w < maxDartboardWidth; w++) {
                    for (int h = 0; h < maxDartboardWidth; h++) {
                        if (y >= 0 && x >= 0 && y < frame.rows && x < frame.cols) {
                            const int loc[4] = {y, x, w, h};
                            featureAccumulator.at<uchar>(loc) = 0;
                        }
                    }
                }
            }
        }
        
        cout << "Dartboard found: maxval: " << maxVal << " maxy: " << maxIdx[0] << " maxx: " << maxIdx[1] << " maxw: " << maxIdx[2] << " maxh: " << maxIdx[3] << endl << flush;
        minMaxIdx(featureAccumulator, &minVal, &maxVal, minIdx, maxIdx);
    }

    cout << "Done" << endl << flush;
}

// Increments a value to a limit, at which point it is capped
int incrementLimit(int current, int incrementBy, int limit) {
    int incremented = current + incrementBy;
    if (incremented > limit) return limit;
    else return incremented;
}
