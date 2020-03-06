// EE4208 Intelligent Systems Design Assignment 2
// Canny Edge Detection Algorithm

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

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
Mat reflectionPadding(Mat image, vector<vector<double> > kernel);	// Padding original image by reflection to avoid border problem after kernel convolution
double createGaussianKernel(vector<vector<double> > & kernel, double sigma);	// Generates a 2D Gaussian Kernel 
void convolve2d(Mat inputImg, Mat outputImg, vector<vector<double> > kernel, double kernWeight);	// 2D Convolution with a 5 X 5 Kernel
int xGradient(Mat outputImg, int x, int y);
int yGradient(Mat outputImg, int x, int y);
void edgeDetect(Mat inputImg, Mat outputImg);
void nonMaxSup();	// Non-Maximum Suppression
void nonMaxSup(Mat inputImg, Mat outputImg);	// Non-Maximum Suppression

constexpr auto PI = 3.14159265359;

// Declaring dimensions of input images
int rowCana = 512, colCana = 479;
int rowFruit = 487, colFruit = 414;
int rowImg335 = 500, colImg335 = 335;
int rowLamp = 256, colLamp = 256;
int rowLeaf = 190, colLeaf = 243;

int main() {
	
	Mat imgMat; // Image produced after conversion from .RAW file
	Mat imgPadded; // Image produced after reflection padding
	Mat imgSmooth; // Smoothed image after convolution with Gaussian Kernel
	Mat imgEdge; // Image with edges detected
	double kernelWeight;

	// To use Cana.raw image
	const char* fileName = "D:\\cana.raw";
	imgMat = convertBinToMat(fileName, colCana, rowCana);

	// Uncomment to use Fruit.raw image
	/*const char* fileName = "D:\\fruit.raw";
	imgMat = convertBinToMat(fileName, colFruit, rowFruit);*/
	
	// Uncomment to use Img335.raw image
	/*const char* fileName = "D:\\img335.raw";
	imgMat = convertBinToMat(fileName, colImg335, rowImg335);*/
	
	// Uncomment to use Lamp.raw image
	/*const char* fileName = "D:\\lamp.raw";
	imgMat = convertBinToMat(fileName, colLamp, rowLamp);*/
	
	// Uncomment to use Leaf.raw image
	/*const char* fileName = "D:\\leaf.raw";
	imgMat = convertBinToMat(fileName, colLeaf, rowLeaf);*/

	imshow("Original Image", imgMat); // display original image

	// Noise Smoothing //
	cout << "Enter a Standard Deviation of your choice: ";
	double sigma;
	cin >> sigma;
	
	vector<vector<double> > gaussianKernel;

	kernelWeight = createGaussianKernel(gaussianKernel, sigma);
	//cout << kernelIndex << endl;

	cout << "\nGaussian Kernel: " << endl;

	for (int i = 0; i < gaussianKernel.size(); ++i)
	{
		for (int j = 0; j < gaussianKernel[i].size(); ++j)
			cout << setprecision(3) << gaussianKernel[i][j] << "\t";
			cout << endl;
	}

	imgPadded = reflectionPadding(imgMat, gaussianKernel); // padding original image to avoid border problem
	imshow("Padded Image", imgPadded); // for debugging
	//cout << "No. of Rows of Padded Image: " << imgPadded.rows << endl; // for debugging
	//cout << "No. of Columns of Padded Image: " << imgPadded.cols << endl; // for debugging

	imgSmooth = imgPadded.clone();
	int index1 = (gaussianKernel.size() - 1) / 2;
	for (int y = index1; y < imgPadded.rows - index1; y++)
		for (int x = index1; x < imgPadded.cols - index1; x++)
			imgSmooth.at<uchar>(y, x) = 0.0;

	convolve2d(imgPadded, imgSmooth, gaussianKernel, kernelWeight);
	imshow("Smoothed Image", imgSmooth); // for debugging
	//cout << "\nNo. of Rows of Smoothed Image: " << imgSmooth.rows << endl; // for debugging
	//cout << "No. of Columns of Smoothed Image: " << imgSmooth.cols << endl; // for debugging

	// Edge Detection //
	Mat tempImg, temp2Img;
	imgSmooth.convertTo(tempImg, CV_32SC1, 255); // tempImg is equivalent to imgSmooth just that it is 32-bit signed
	temp2Img = tempImg.clone();
	int index2 = (gaussianKernel.size() - 1) / 2;
	for (int y = index2; y < tempImg.rows - index2; y++)
		for (int x = index2; x < tempImg.cols - index2; x++)
			temp2Img.at<int>(y, x) = 0.0;

	edgeDetect(imgSmooth, temp2Img);
	temp2Img.convertTo(imgEdge, CV_8UC1, 2); // temp2Img is equivalent to imgEdge just that it is 32-bit signed
	imshow("Image Edges", imgEdge);

	// Edge Enhancement by Non-Maximum Suppression //

	Mat temp3Img;
	temp3Img = temp2Img.clone(); // temp2Img is equivalent to imgEdge just that it is 32-bit signed
	int index3 = (gaussianKernel.size() - 1) / 2;
	for (int y = index3; y < temp2Img.rows - index3; y++)
		for (int x = index3; x < temp2Img.cols - index3; x++)
			temp3Img.at<int>(y, x) = 0.0;

	nonMaxSup(temp2Img, temp3Img);
	temp3Img.convertTo(imgThin, CV_8UC1, 2); // temp3Img is equivalent to imgThin just that it is 32-bit signed
	imshow("Thinned Image", imgThin);

	waitKey(0);

return 0;
}

// Function Definitions

// Convert .RAW binary data into a matrix
Mat convertBinToMat(const char* fileName, int col, int row) {
	ifstream input(fileName, ios::binary);
	vector<uchar> originalBuffer(istreambuf_iterator<char>(input), {});
	vector<uchar> buffer(originalBuffer.begin(), originalBuffer.end());
	Mat image = Mat(col, row, CV_8UC1);
	memcpy(image.data, buffer.data(), buffer.size() * sizeof(unsigned char));
	//imshow("Original Image", image); // for debugging

	cout << "\n" << fileName << " has been successfully loaded!\n" << endl;

	return image;
}

// Padding input image by reflection so that original image dimensions are retained after kernel convolution
Mat reflectionPadding(Mat image, vector<vector<double> > kernel) {
	int border = (kernel.size() - 1) / 2;
	Mat padded(image.rows + border * 2, image.cols + border * 2, image.depth()); // constructs a larger image to fit both the image and the border
	copyMakeBorder(image, padded, border, border, border, border, BORDER_REPLICATE); // form a border in-place
	//imshow("Padded Image", padded); // for debugging

	cout << "\nReflection Padding Successful!" << endl;

	return padded;
}

// Gaussian Formula
double gaussian(int x, int y, double sigma) {
	return (1 / (2 * PI * pow(sigma, 2))) * exp(-1 * (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
}

// Generating a 2D Gaussian Kernel to smooth the image vector
double createGaussianKernel(vector<vector<double> > & kernel, double sigma) {
	int kernRow = 0, kernIndex = 0;
	double kernWeight = 0;
	if (fmod(sigma, 2) == 0) { // checking if sigma is an even or odd number
		kernRow = 5 * sigma + 1; // sigma is even
		cout << "Standard Deviation Entered: sigma = " << sigma << " (EVEN)\n" << endl;
		//cout << "Sigma = " << sigma << " is EVEN" << endl;
		cout << "Kernel Size: " << kernRow << " X " << kernRow << endl;
		kernIndex = (kernRow - 1) / 2;
		cout << "Kernal Indexing from: " << -1 * kernIndex << " to " << kernIndex << endl;
	}
	else {
		kernRow = 5* sigma; // sigma is odd
		cout << "Standard Deviation Entered: sigma = " << sigma << " (ODD)\n" << endl;
		//cout << "Sigma = " << sigma << " is ODD" << endl;
		cout << "Kernel Size: " << kernRow << " X " << kernRow << endl;
		kernIndex = (kernRow - 1) / 2;
		cout << "Kernel Indexing from: " << -1 * kernIndex << " to " << kernIndex << endl;
	}
	
	double smallest = gaussian(-1 * kernIndex, -1 * kernIndex, sigma);
	//cout << "Gaussian: " << gaussian << endl; // for debugging

	for (int i = -1 * kernIndex; i <= kernIndex; i++) {
		vector<double> temp;
		for (int j = -1 * kernIndex; j <= kernIndex; j++) {
			int gVal = round(gaussian(i, j, sigma) / smallest);
			temp.push_back(gVal);
			kernWeight += gVal;
		}
		kernel.push_back(temp);
	}
	
	cout << "Weighted Sum: " << kernWeight << endl;

	return kernWeight;
}

// 2D Convolution with a 5 X 5 Kernel
void convolve2d(Mat inputImg, Mat outputImg, vector<vector<double> > kernel, double kernWeight) {
	inputImg.convertTo(outputImg, CV_8UC1); // convert inputImg into type CV_8UC1 (8bit, single channel) and store into outputImg
	//float row = 2, col = 2; // for indexing inputImg
	int kernIndex = (kernel.size() - 1) / 2; // getting the kernel index from the size of kernel
	//cout << "kernIndex: " << kernIndex << endl; // for debugging
	for (int row = kernIndex; row < inputImg.rows - kernIndex; row++) {		// inputImg rows
		for (int col = kernIndex; col < inputImg.cols - kernIndex; col++) {	// inputImg columns
			double weightedSum = 0;
			for (int i = -1 * kernIndex; i <= kernIndex; i++) {		// kernel rows
				for (int j = -1 * kernIndex; j <= kernIndex; j++) {	// kernel columns
					weightedSum += kernel[i + kernIndex][j + kernIndex] * inputImg.at<uchar>(row + i, col + j); // single pixel convolution
				}
			}
			outputImg.at<uchar>(row, col) = weightedSum/kernWeight; // assign to the pixel the weighted sum of its neighbouring pixels
		}
	}
}

// Calculate gradient intensity of image vector
// Apply Sobel Convolution Masks in X and Y directions

// 1D Sobel Operator in X direction
int xGradient(Mat outputImg, int x, int y) {
	return	outputImg.at<uchar>(y - 1, x - 1) +
			2 * outputImg.at<uchar>(y, x - 1) +
			outputImg.at<uchar>(y + 1, x - 1) -
			outputImg.at<uchar>(y - 1, x + 1) -
			2 * outputImg.at<uchar>(y, x + 1) -
			outputImg.at<uchar>(y + 1, x + 1);
}

// 1D Sobel Operator in Y direction
int yGradient(Mat outputImg, int x, int y) {
	return	outputImg.at<uchar>(y - 1, x - 1) +
			2 * outputImg.at<uchar>(y - 1, x) +
			outputImg.at<uchar>(y - 1, x + 1) -
			outputImg.at<uchar>(y + 1, x - 1) -
			2 * outputImg.at<uchar>(y + 1, x) -
			outputImg.at<uchar>(y + 1, x + 1);
}

// Detects edges in the input image using two 1D Sobel Operators
void edgeDetect(Mat inputImg, Mat outputImg) {
	int fx, fy, magnitude, magSum;
	float up = 0, down = 0, left = 0, right = 0;
	float dir, dirSum;
	for (int y = 1; y < inputImg.rows - 1; y++) {
		for (int x = 1; x < inputImg.cols - 1; x++) {
			fx = xGradient(inputImg, x, y);
			fy = yGradient(inputImg, x, y);
			
			// this is the actual formula but it requires high comuting power to evaluate
			//magnitude = sqrt(pow(fx, 2) + pow(fy, 2));
			// thus the formula in the next line is used which gives an accurate approximation
			magnitude = abs(fx) + abs(fy); // gradient magnitudes
			
			dir = atan2(fy, fx) * 180 / PI; // gradient direction in degrees
			
			if (dir == 0.0 || dir == 360.0)
				up++;
			else if (dir == 90.0)
				right++;
			else if (dir == 180.0)
				down++;
			else if (dir == 270.0)
				left++;

			outputImg.at<int>(y, x) = magnitude;
		}
	}
	cout << "Up: " << up << endl;
	cout << "Down: " << down << endl;
	cout << "Left: " << left << endl;
	cout << "Right: " << right << endl;
}

// Find gradient strength and direction


// Thinning Edges by Non-Maximum Supression // 
void nonMaxSup(Mat inputImg, Mat outputImg) {
	int fx, fy;
	for (int y = 1; y < inputImg.rows - 1; y++) {
		for (int x = 1; x < inputImg.cols - 1; x++) {
			fx = xGradient(inputImg, x, y);
			fy = yGradient(inputImg, x, y);
			if (abs(fx) > abs(fy)) { // horizontal edge
				int top = abs(xGradient(inputImg, x - 1, y)) + abs(yGradient(inputImg, x - 1, y));
				int center = abs(xGradient(inputImg, x, y)) + abs(yGradient(inputImg, x, y));
				int bottom = abs(xGradient(inputImg, x + 1, y)) + abs(yGradient(inputImg, x + 1, y));
				if (center > top && center > bottom) { // center pixel has highest magnitude
					outputImg.at<int>(y, x) = center; // assign center pixel its edge magnitude
				}
				else // center pixel is not highest magnitude
					outputImg.at<int>(y, x) = 0; // suppress center pixel
			}
			else if (abs(fx) < abs(fy)) { // vertical edge
				int left = abs(xGradient(inputImg, x, y - 1)) + abs(yGradient(inputImg, x, y - 1));
				int center = abs(xGradient(inputImg, x, y)) + abs(yGradient(inputImg, x, y));
				int right = abs(xGradient(inputImg, x, y + 1)) + abs(yGradient(inputImg, x, y + 1));
				if (center > left && center > right) { // center pixel has highest magnitude
					outputImg.at<int>(y, x) = center; // assign center pixel its edge magnitude
				}
				else // center pixel is not highest magnitude
					outputImg.at<int>(y, x) = 0; // suppress center pixel
			}
			else if ((fx * fy) < 0) { // oblique edge: top left to bottom right
				int topleft = abs(xGradient(inputImg, x - 1, y - 1)) + abs(yGradient(inputImg, x - 1, y - 1));
				int center = abs(xGradient(inputImg, x, y)) + abs(yGradient(inputImg, x, y));
				int bottomright = abs(xGradient(inputImg, x + 1, y + 1)) + abs(yGradient(inputImg, x + 1, y + 1));
				if (center > topleft && center > bottomright) { // center pixel has highest magnitude
					outputImg.at<int>(y, x) = center; // assign center pixel its edge magnitude
				}
				else // center pixel is not highest magnitude
					outputImg.at<int>(y, x) = 0; // suppress center pixel
			}
			else if ((fx * fy) > 0) { // oblique edge: bottom left to top right
				int bottomleft = abs(xGradient(inputImg, x + 1, y - 1)) + abs(yGradient(inputImg, x + 1, y - 1));
				int center = abs(xGradient(inputImg, x, y)) + abs(yGradient(inputImg, x, y));
				int topright = abs(xGradient(inputImg, x - 1, y + 1)) + abs(yGradient(inputImg, x - 1, y + 1));
				if (center > bottomleft && center > topright) { // center pixel has highest magnitude
					outputImg.at<int>(y, x) = center; // assign center pixel its edge magnitude
				}
				else // center pixel is not highest magnitude
					outputImg.at<int>(y, x) = 0; // suppress center pixel
			}
		}
	}
}
