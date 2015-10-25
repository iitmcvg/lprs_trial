#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

using namespace cv;
using namespace std;

int EXIT_FAILURE = 1;
int MODE_IMAGE = 0;
int MODE_VIDEO = 1;

int main(int argc, char** argv) {
	if(argc < 3) {
		cout<<"Usage : "<<argv[0]<<" <mode> <image/video>"<<endl;
		exit(EXIT_FAILURE);
	}

	CascadeClassifier licensePlateCascade;
	string location = "../xmls/haarcascade_russian_plate_number.xml";
	licensePlateCascade.load(location);

	if(mode == MODE_IMAGE) {
		Mat img = imread(argv[2], 0);

		vector<Rect> plates;
		licensePlateCascade.detectMultiScale( img, plates, 1.3, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
		
	}
}