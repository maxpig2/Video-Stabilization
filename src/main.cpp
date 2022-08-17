
// std
#include <iostream>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//Ptr<SIFT> s = SIFT::create();

Mat_<Vec3b> inputArray[101];


inline double getErr(Point2f p1, Point2f p2, Mat_<double> H) {
	double z = (H[2][0] * p1.x + H[2][1] * p1.y + H[2][2]);
	double x = (H[0][0] * p1.x + H[0][1] * p1.y + H[0][2]) / z;
	double y = (H[1][0] * p1.x + H[1][1] * p1.y + H[1][2]) / z;
	return (x - p2.x) * (x - p2.x) + (y - p2.y) * (y - p2.y);
}

void display(string img_display, Mat m) {
	namedWindow(img_display, WINDOW_AUTOSIZE);
	imshow(img_display, m);
	waitKey(10);
}

void loadImages(String files) {

	int frameVal = 0;
	// parse all images
	cout << "Loading input array ..." << std::endl;
	vector<String> lf_imgs;
	glob(files, lf_imgs);
	for (String cv_str : lf_imgs) {
		// get the filepath
		string filepath(cv_str);
		size_t pos = filepath.find_last_of("/\\");
		if (pos != std::string::npos) {
			string filename = filepath.substr(pos + 1);
			// parse for values
			istringstream ss(filename);
			string name;
			if (ss.good()) {
				cout << "Image loaded" << frameVal << endl;
				inputArray[frameVal] = imread(filepath, 1);
				frameVal++;
				continue;

			}
		}
		// throw error otherwise
		std::cerr << "Filepath error with : " << filepath << std::endl;
		abort();

	}
	std::cout << "Finished loading input array" << std::endl;
}

Mat_<Vec3b> getInputImage(int i) {
	if (i < 0 || i > 101) {
		cerr << "InputImageValue not avaliable : " << i << endl;
	}
	return inputArray[i];
}

//Ransac
Mat ransac(int size, vector<Point> query, vector<Point> train, double e) {
	int bestInliers = -1;
	double epsilon = e;
	Mat bestHomography;
	//float best_error = numeric_limits<float>::infinity();
	int iterations = 100;
	//Start of iteration
	for (int i = 0; i < iterations; i++) {
		//Choose 4 different pairs of points
		int index[4];
		vector<Point> choice1;
		vector<Point> choice2;
		for (int j = 0; j < 4; j++) {
			bool duplicate;
			do {
				index[j] = rand() % size;
				duplicate = false;
				for (int k = 0; k < j; k++) {
					if (index[k] == index[j]) {
						duplicate = true;
					}

				}
			} while (duplicate);
			choice1.push_back(query.at(index[j]));
			choice2.push_back(train.at(index[j]));
		}
		//computeHomography transfomration for the pairs
		Mat newHomography = findHomography(choice1, choice2);
		int inliers = 0;
		for (int j = 0; j < size; j++) {
			//Compute inliers amongst all of the pairs
			double err = getErr(query[j], train[j], newHomography);
			if (err <= epsilon) {
				inliers++;
			}
		}
		//Save best homography if there are more inliers present
		if (inliers > bestInliers) {
			bestInliers = inliers;
			bestHomography = newHomography;
		}
	}

	vector<Point> goodQuery;
	vector<Point> goodTrain;

	bestInliers = 0;
	//Re-compute homography transform H on largest set of inlier pairs
	for (int j = 0; j < size; j++) {
		double err = getErr(query[j], train[j], bestHomography);
		cout << "Err" << err << endl;
		if (err <= epsilon) {
			goodQuery.push_back(query[j]);
			goodTrain.push_back(train[j]);
			bestInliers++;
		}
	}


	Mat finalHomography = findHomography(goodQuery, goodTrain);
	return finalHomography;
}

void videoStabilizer() {
	bool saveImages = true;
	int nframes = 102;
	Ptr<SIFT> sift = SIFT::create();
	Ptr<FeatureDetector> detector = SIFT::create(1000);
	Mat_<uint8_t> mask;
	int rows = inputArray[0].rows;
	int cols = inputArray[0].cols;
	mask = Mat_<uint8_t>(rows, cols);
	mask = 255;
	Ptr<DescriptorMatcher> bfm = BFMatcher::create(NORM_L2, true);
	// ~
	// H
	Mat_<double> cumulativeHomography = Mat_<double>::eye(3, 3);
	//Homography of each frame
	vector<Mat_<double>> frameHomography;
	const int filterSize = 9;
	//Gaussian Filter parameters
	float filterWeights[filterSize + 1];
	float sigma = 2.0;
	for (int i = 0; i < filterSize + 1; i++) {
		filterWeights[i] = exp(-(i * i) / (2 * sigma * sigma));
		cout << i << " " << filterWeights[i] << endl;
	}

	Mat previousDescriptor;
	vector<KeyPoint> previousKeyPoints;
	vector<DMatch> dm;
	detector->detectAndCompute(inputArray[0], mask, previousKeyPoints, previousDescriptor);


	//Go through each frame
	for (int frame = 0; frame < nframes + filterSize; frame++) {
		if (frame < nframes) {
			Mat descriptor;
			vector<KeyPoint> keyPoints;
			detector->detectAndCompute(inputArray[frame], mask, keyPoints, descriptor);
			bfm->match(descriptor, previousDescriptor, dm);
			vector<Point> goodQuery;
			vector<Point> goodTrain;

			for (DMatch d : dm) {
				goodQuery.push_back(keyPoints[d.queryIdx].pt);
				goodTrain.push_back(previousKeyPoints[d.trainIdx].pt);
			}

			double epsilon = 100.0;
			int size = dm.size();
			Mat homography = ransac(size,goodQuery,goodTrain,epsilon);
			cumulativeHomography *= homography;
			frameHomography.push_back(cumulativeHomography.clone());
			previousKeyPoints = keyPoints;
			previousDescriptor = descriptor;
		}
		//Smooth using gaussian filter
		if (frame >= filterSize) {
			Mat_<double> smooth = Mat_<double>::zeros(3, 3);
			double s = 0;
			for (int i = -filterSize; i <= filterSize; i++) {
				int si = clamp(frame - filterSize + i, 0, nframes - 1);
				double w = filterWeights[abs(i)];
				smooth += frameHomography[si] * w;
				s = s + w;
			}
			smooth = smooth * (1 / s);
			Mat_<Vec3b> transformed(rows, cols);

			Mat transform = smooth.inv() * frameHomography[frame - filterSize];
			warpPerspective(inputArray[frame - filterSize], transformed, transform, Size(transformed.cols, transformed.rows), INTER_LINEAR, BORDER_CONSTANT, Vec3b(0, 255, 0));

			if (saveImages) {
				std::stringstream ss;
				ss << "Stable" << std::setw(3) << std::setfill('0') << (frame - filterSize) << ".png";
				imwrite(ss.str(), transformed);
			}


		}


	}



}


void featurePoints(int in1, int in2) {
	Mat descriptor1, descriptor2;
	Ptr<SIFT> sift = SIFT::create();
	Ptr<FeatureDetector> detector = SIFT::create(1000);
	Mat_<uint8_t> mask;
	int rows = inputArray[0].rows;
	int cols = inputArray[0].cols;
	mask = Mat_<uint8_t>(rows, cols);
	mask = 255;
	vector<KeyPoint> input1, input2;
	vector<DMatch> dm;
	detector->detectAndCompute(inputArray[in1], mask, input1, descriptor1);
	detector->detectAndCompute(inputArray[in2], mask, input2, descriptor2);
	Ptr<DescriptorMatcher> bfm = BFMatcher::create(NORM_L2, true);
	bfm->match(descriptor1, descriptor2, dm);

	Mat_<Vec3b> stackedImage(rows * 2, cols);
	vconcat(inputArray[in1], inputArray[in2], stackedImage);
	Scalar col(0, 255, 0);
	vector<Point> query;
	vector<Point> train;

	

	for (DMatch d : dm) {
		Point p1 = input1[d.queryIdx].pt;
		Point p2 = input2[d.trainIdx].pt;

		query.push_back(p1);
		train.push_back(p2);

		p2.y += rows;
		line(stackedImage, p1, p2, col);
	}
	display("Connected Feature Points", stackedImage);

	int size = dm.size();
	double epsilon = 100.0;
	Mat_<double> ransacHomography = ransac(size, query, train, epsilon);
	
	for (int j = 0; j < size; j++) {
		double err = -1;
		Scalar col(0, 0, 255);
		if (!ransacHomography.empty())
			err = getErr(query[j], train[j], ransacHomography);
		if (err <= epsilon) {
		col = Scalar(0, 255, 0);
			
		}
		Point p1 = query[j];
		Point p2 = train[j];
		p2.y += rows;
		line(stackedImage, p1, p2, col);
	}
	

	display("Inliers", stackedImage);

	int sizeIncrease = 64;
	rows = inputArray[in1].rows;
	cols = inputArray[in1].cols;
	//Warping the Image
	Mat_<Vec3b> warp(rows + sizeIncrease, cols + sizeIncrease);
	//Adjust x and y to account for size increase
	ransacHomography(1, 2) += sizeIncrease/2;
	ransacHomography(0, 2) += sizeIncrease/2;
	//Warp Image
	warpPerspective(inputArray[in1], warp, ransacHomography, Size(warp.cols, warp.rows), INTER_LINEAR, BORDER_CONSTANT, Vec3b(0, 255, 0));
	for (int i = 0; i < rows; i++) {
		 for (int j = 0; j < cols; j++) {
			warp(i + sizeIncrease / 2, j + sizeIncrease / 2) = inputArray[in2](i, j);
			// use the following to average the images
				//warp(i+ sizeIncrease / 2,j+ sizeIncrease / 2)=Vec3b((Vec3i(warp(i+ sizeIncrease / 2,j+ sizeIncrease / 2))+Vec3i(inputArray[in2](i,j)))/2);
		}
	}
	
	display("Warped Image", warp);

	
}



// main program
// 
int main( int argc, char** argv ) {

	loadImages("res/input/*.jpg");
	//display("Test", getInputImage(101));
	featurePoints(39, 41);

	cout << endl;
	videoStabilizer();
	
	// wait for a keystroke in the window before exiting
	waitKey(0);
}
