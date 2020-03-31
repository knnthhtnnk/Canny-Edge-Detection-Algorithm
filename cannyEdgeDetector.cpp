// EE4208 Intelligent Systems Design Assignment 2
// Canny Edge Detection Algorithm

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

using namespace cv;
using namespace std;

// Function Declarations
Mat convertBinToMat(const char* fileName, int col, int row);	// Convert .RAW binary data into a matrix
Mat reflectionPadding(Mat src, vector<vector<double> > kernel);	// Padding original image by reflection to avoid border problem after kernel convolution
double gaussian(int x, int y, double sigma); // Gaussian Formula
double createGaussianKernel(vector<vector<double> >& kernel, double sigma);	// Generates a 2D Gaussian Kernel
Mat initMat(Mat src); // Initialise dst image to be same size as src image but populated with zeros
Mat convolve2d(Mat src, vector<vector<double> > kernel, bool type); // convolution
Mat edges(Mat fx, Mat fy); // generates edge map
Mat nonMaxSup(Mat fx, Mat fy, Mat edges);
Mat hysteresis(Mat src, int minThresh, int maxThresh);

constexpr auto PI = 3.14159265359;

// Declaring dimensions of input images
int rowCana = 512, colCana = 479;
int rowFruit = 487, colFruit = 414;
int rowImg335 = 500, colImg335 = 335;
int rowLamp = 256, colLamp = 256;
int rowLeaf = 190, colLeaf = 243;

int main() {
	const char* cana = "D:\\cana.raw";
	const char* fruit = "D:\\fruit.raw";
	const char* img335 = "D:\\img335.raw";
	const char* lamp = "D:\\lamp.raw";
	const char* leaf = "D:\\leaf.raw";

	Mat imgMat;

	bool valid;

	do {
		cout << "\nWhich image do you want to use?\n" << endl;
		cout << "1: Cana.raw \n2: Fruit.raw \n3: Img335.raw \n4: Lamp.raw \n5: Leaf.raw" << endl;
		cout << "\nPlease enter your choice: ";
		int choice;
		cin >> choice;
		
		switch (choice) {
		case 1:
			imgMat = convertBinToMat(cana, colCana, rowCana);
			valid = true;
			break;
		case 2:
			imgMat = convertBinToMat(fruit, colFruit, rowFruit);
			valid = true;
			break;
		case 3:
			imgMat = convertBinToMat(img335, colImg335, rowImg335);
			valid = true;
			break;
		case 4:
			imgMat = convertBinToMat(lamp, colLamp, rowLamp);
			valid = true;
			break;
		case 5:
			imgMat = convertBinToMat(leaf, colLeaf, rowLeaf);
			valid = true;
			break;
		default:
			cout << "\nWrong Input!\nPlease only enter an INTEGER from 1 to 5!" << endl;
			valid = false;
			break;
		}
		cin.clear(); // clear the cin error flags
		cin.ignore(); // clear the cin buffer
	} while (!valid);
		
	// Noise Smoothing with Gaussian Kernel //
	cout << "Enter a Standard Deviation of your choice: ";
	double sigma;
	cin >> sigma;

	imshow("Original Image", imgMat); // display original image

	vector<vector<double> > gaussianKernel; // Gaussian Kernel
	createGaussianKernel(gaussianKernel, sigma);

	cout << "\nGaussian Kernel: " << endl;

	for (int i = 0; i < gaussianKernel.size(); ++i) {
		for (int j = 0; j < gaussianKernel[i].size(); ++j) {
			cout << setprecision(3) << gaussianKernel[i][j] << "\t";
		}
		cout << endl;
	}

	Mat imgPadded = reflectionPadding(imgMat, gaussianKernel); // padding original image to avoid border problem
	//imshow("Padded Image", imgPadded); // for debugging

	Mat imgSmooth = initMat(imgPadded); // initialising smoothed image populated with zeros
	imgSmooth = convolve2d(imgPadded, gaussianKernel, 0); // imgSmooth is a 8-bit unsigned Mat
	//imshow("Smoothed Image", imgSmooth); // for debugging
	
	// 1D Sobel Kernels
	/*vector<vector<double> > sobelXkernel{ {-1},{0},{1} };
	vector<vector<double> > sobelYkernel{ {-1,0,1} };*/

	// 2D Sobel Kernels
	vector<vector<double> > sobelXkernel{	{-1, 0, 1},
											{-2, 0, 2},
											{-1, 0, 1} };
	vector<vector<double> > sobelYkernel{	{1, 2, 1},
											{0, 0, 0},
											{-1, -2, -1} };
	
	// 2D Prewitt Kernels
	/*vector<vector<double> > sobelXkernel{	{-1, 0, 1},
											{-1, 0, 1},
											{-1, 0, 1} };
	vector<vector<double> > sobelYkernel{	{1, 1, 1},
											{0, 0, 0},
											{-1, -1, -1} };*/
	
	Mat fx = initMat(imgSmooth);
	Mat fy = initMat(imgSmooth);

	// Convoluting smoothed image with horizontal & vertical Sobel Kernels //
	fx = convolve2d(imgSmooth, sobelXkernel, 1); // fx is a 32-bit signed horizontal derivative of imgSmooth
	fy = convolve2d(imgSmooth, sobelYkernel, 1); // fy is a 32-bit signed vertical derivative of imgSmooth
	//Mat Fx, Fy;
	//fx.convertTo(Fx, CV_8UC1); // convert to 8-bit unsigned to display
	//fy.convertTo(Fy, CV_8UC1); // convert to 8-bit unsigned to display
	//imshow("Fx", Fx);
	//imshow("Fy", Fy);

	// Computing Edge Map // 
	Mat imgEdge = edges(fx, fy);
	imgEdge.convertTo(imgEdge, CV_8UC1);
	//imshow("Image Edge Map", imgEdge);

	// Non-Maximum Suppression //
	Mat imgThin = nonMaxSup(fx, fy, imgEdge);
	imshow("Thinned Image", imgThin);

	// Hysteresis Thresholding //
	int minThresh = 20; // minimum pixel value to be considered an edge
	int maxThresh = 180; // maximum pixel value to be considered an edge
	Mat imgHyst = hysteresis(imgThin, minThresh, maxThresh);
	imshow("Image after Hysteresis", imgHyst);

	waitKey(0);
	return 0;
}

// Function Definitions

// Convert .RAW binary data into a matrix
Mat convertBinToMat(const char* fileName, int col, int row) {
	ifstream input(fileName, ios::binary);
	vector<uchar> originalBuffer(istreambuf_iterator<char>(input), {});
	vector<uchar> buffer(originalBuffer.begin(), originalBuffer.end());
	Mat image = Mat(col, row, CV_8UC1); // 8 bit unsigned single cahnnel image
	memcpy(image.data, buffer.data(), buffer.size() * sizeof(unsigned char));
	//imshow("Original Image", image); // for debugging

	cout << "\n" << fileName << " has been successfully loaded!\n" << endl;

	return image;
}

// Gaussian Formula
double gaussian(int x, int y, double sigma) {
	return (1 / (2 * PI * pow(sigma, 2))) * exp(-1 * (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
}

// Generating a 2D Gaussian Kernel to smooth the image vector
double createGaussianKernel(vector<vector<double> >& kernel, double sigma) {
	double temp = 0.0, temp2 = 0.0;
	double kernRow = 0.0, kernIndex = 0.0;
	double kernWeight = 0.0;
	if (sigma >= 1) { // checking if sigma is greater than 1
		temp = ceil(sigma); // rounding up to nearest integer
		if (fmod(temp, 2) == 0) { // checking if sigma is an even number
			temp2 = (5 * temp) + 1;
			kernRow = temp2;
		}
		if (fmod(temp, 2) == 1) { // checking if sigma is an odd number
			temp2 = 5 * temp;
			kernRow = temp2;
		}
		cout << "Standard Deviation Entered: sigma = " << sigma << endl;
		cout << "Gaussian Kernel Size: " << kernRow << " X " << kernRow << endl;
	}
	if (sigma < 1) { // checking if sigma is less than 1
		kernRow = 3; // assign smallest kernel size: 3
		cout << "Standard Deviation Entered: sigma = " << sigma << " (NON-integer: less than 1)\n" << endl;
		//cout << "Sigma = " << sigma << " is less than 1" << endl;
		cout << "Gaussian Kernel Size: " << kernRow << " X " << kernRow << endl;
	}

	kernIndex = (kernRow - 1) / 2;
	cout << "Gaussian Kernal Indexing from: " << -1 * kernIndex << " to " << kernIndex << endl;

	double filter = gaussian(-1 * kernIndex, -1 * kernIndex, sigma);
	//cout << "Gaussian: " << gaussian << endl; // for debugging

	for (int row = -1 * kernIndex; row <= kernIndex; row++) {
		vector<double> tempKern;
		for (int col = -1 * kernIndex; col <= kernIndex; col++) {
			int val = round(gaussian(row, col, sigma) / filter);
			tempKern.push_back(val);
			kernWeight += val;
		}
		kernel.push_back(tempKern);
	}

	cout << "Gaussian Kernal Weighted Sum: " << kernWeight << endl;

	return kernWeight;
}

// Padding input image by reflection so that original image dimensions are retained after kernel convolution
Mat reflectionPadding(Mat src, vector<vector<double> > kernel) {
	int border = (kernel.size() - 1) / 2;
	Mat dst(src.rows + border * 2, src.cols + border * 2, src.depth()); // constructs a larger image to fit both the image and the border
	copyMakeBorder(src, dst, border, border, border, border, BORDER_REPLICATE); // form a border in-place
	//imshow("Padded Image", padded); // for debugging

	cout << "\nReflection Padding Successful!" << endl;
	//cout << "No. of Rows of Padded Image: " << dst.rows << endl; // for debugging
	//cout << "No. of Columns of Padded Image: " << dst.cols << endl; // for debugging

	return dst;
}

// Initialises dst image to the same size as src and populated with zeros
Mat initMat(Mat src) {
	Mat dst = src.clone();
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			if (src.type() == 0) { // CV_8UC1
				dst.at<uchar>(y, x) = 0.0;
			}
			if (src.type() == 4) { // CV_32SC1
				dst.at<int>(y, x) = 0.0;
			}
		}
	}
	return dst;
}

// 2D Convolution with specified kernel
Mat convolve2d(Mat src, vector<vector<double> > kernel, bool type) {
	Mat dst; // output image
	int kernXcenter = floor(kernel.size() / 2);
	int kernYcenter = floor(kernel[0].size() / 2);

	if (type) { // checking Mat image type
		//dst = cv::Mat::zeros(src.size(), CV_32SC1); // initialise the dst image as 32-bit signed
		src.convertTo(dst, CV_32SC1);
	}
	else {
		//dst = cv::Mat::zeros(src.size(), CV_8UC1); // initialise the dst image as 8-bit unsigned
		src.convertTo(dst, CV_8UC1);
	}
	for (int row = kernXcenter; row < src.rows - kernXcenter; row++) {
		for (int col = kernYcenter; col < src.cols - kernYcenter; col++) {
			double total = 0;
			double weightedSum = 0;
			for (int i = -1 * kernXcenter; i <= kernXcenter; i++) {
				for (int j = -1 * kernYcenter; j <= kernYcenter; j++) {
					weightedSum += kernel[kernXcenter + i][kernYcenter + j];
					total += kernel[kernXcenter + i][kernYcenter + j] * src.at<uchar>(row + i, col + j);
				}
			}
			if (type)
				dst.at<int>(row, col) = (int)round(total / max(weightedSum, 1.0));
			else
				dst.at<uchar>(row, col) = (uchar)round(total / max(weightedSum, 1.0));
		}
	}

	cout << "\nConvolution Successful!" << endl;

	return dst;
}

// Forms the edge map from x and y Sobel derivatives of an image
Mat edges(Mat fx, Mat fy) {
	Mat dst;
	fx = abs(fx);
	fy = abs(fy);

	fx.convertTo(fx, CV_32F);
	fy.convertTo(fy, CV_32F);
	
	// magnitude = |fx| + |fy|
	dst = fx + fy;

	// magnitude = sqrt(fx^2 + fy^2)
	/*add(fx.mul(fx), fy.mul(fy), dst);
	sqrt(dst, dst);*/
	
	cout << "\nEdge Map Produced Successfully!" << endl;

	return dst; // dst is a 32-bit signed Mat
}

Mat nonMaxSup(Mat fx, Mat fy, Mat edges) {
	Mat horizontal, vertical, oblique1, oblique2;

	subtract(abs(fx), abs(fy), horizontal);
	subtract(horizontal, 100, horizontal);
	horizontal = max(horizontal, 0);
	horizontal.convertTo(horizontal, CV_8UC1);

	for (size_t x = 1; x < edges.rows - 1; x++) {
		for (size_t y = 0; y < edges.cols; y++) {
			if ((edges.at<uchar>(x, y) > edges.at<uchar>(x - 1, y) && edges.at<uchar>(x, y) > edges.at<uchar>(x + 1, y))) {
				horizontal.at<uchar>(x, y) = edges.at<uchar>(x, y);
			}
			else
				horizontal.at<uchar>(x, y) = 0;
		}
	}

	subtract(abs(fy), abs(fx), vertical);
	subtract(vertical, 100, vertical);
	vertical = max(vertical, 0);
	vertical.convertTo(vertical, CV_8UC1);

	for (size_t x = 0; x < edges.rows; x++) {
		for (size_t y = 1; y < edges.cols - 1; y++) {
			if ((edges.at<uchar>(x, y) > edges.at<uchar>(x, y - 1) && edges.at<uchar>(x, y) > edges.at<uchar>(x, y + 1))) {
				vertical.at<uchar>(x, y) = edges.at<uchar>(x, y);
			}
			else
				vertical.at<uchar>(x, y) = 0;
		}
	}

	multiply(fx, fy, oblique1);
	oblique1 = max(-1 * oblique1, 0);
	oblique1.convertTo(oblique1, CV_8UC1);

	for (size_t x = 1; x < edges.rows - 1; x++) {
		for (size_t y = 1; y < edges.cols - 1; y++) {
			if ((edges.at<uchar>(x, y) > edges.at<uchar>(x + 1, y - 1) && edges.at<uchar>(x, y) > edges.at<uchar>(x - 1, y + 1))) {
				oblique1.at<uchar>(x, y) = edges.at<uchar>(x, y);
			}
			else
				oblique1.at<uchar>(x, y) = 0;
		}
	}

	multiply(fx, fy, oblique2);
	oblique2 = max(oblique2, 0);
	oblique2.convertTo(oblique2, CV_8UC1);

	for (size_t x = 1; x < edges.rows - 1; x++) {
		for (size_t y = 1; y < edges.cols - 1; y++) {
			if ((edges.at<uchar>(x, y) > edges.at<uchar>(x - 1, y - 1) && edges.at<uchar>(x, y) > edges.at<uchar>(x + 1, y + 1))) {
				oblique2.at<uchar>(x, y) = edges.at<uchar>(x, y);
			}
			else
				oblique2.at<uchar>(x, y) = 0;
		}
	}

	/*thin(edges, horizontal, 1);
	thin(edges, vertical, 2);
	thin(edges, oblique1, 3);
	thin(edges, oblique2, 4);*/

	Mat dst = horizontal + vertical + oblique1 + oblique2;
	dst.convertTo(dst, CV_8UC1);

	cout << "\nNon-Maximum Suppression Successful!" << endl;

	return dst;
}

Mat hysteresis(Mat src, int minThresh, int maxThresh) {
	Mat dst = initMat(src);
	for (int row = 0; row <= src.rows - 1; row++) {
		for (int col = 0; col <= src.cols - 1; col++) {
			// less than minimum threshold: non-edge (discarded)
			if (src.at<uchar>(row, col) < minThresh) {
				dst.at<uchar>(row, col) = 0;
			}
			// between min & max threshold: retained if connected to sure-edge, otherwise discarded
			/*if (src.at<uchar>(row, col) >= minThresh && src.at<uchar>(row, col) <= maxThresh) {

			}*/
			// more than maximum threshold: sure-edge (retained)
			if (src.at<uchar>(row, col) > maxThresh) {
				dst.at<uchar>(row, col) = 255;
			}
		}
	}
	return dst;
}
