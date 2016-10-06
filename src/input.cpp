/*
 * input.cpp
 *
 *  Created on: Sep 9, 2016
 *      Author: xwong
 */

#include "input.h"

input::input() {
	// TODO Auto-generated constructor stub
	fileExtension = std::string("pgm");
	sequenceCounter = 0;
}

bool input::update(cv::Mat& leftImg, cv::Mat& rightImg) {

	/// update file name with sequenceCounter
	char LeftFile[256];
	char RightFile[256];

	std::sprintf(LeftFile,"Left%d.",sequenceCounter);
	std::sprintf(RightFile,"Right%d.",sequenceCounter);

	std::string LFileName(LeftFile);
	std::string RFileName(RightFile);

	LFileName += fileExtension;
	RFileName += fileExtension;

	cv::Mat Limg = cv::imread(LFileName,CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat Rimg = cv::imread(RFileName,CV_LOAD_IMAGE_GRAYSCALE);

//	std::cout<<"reading image : "<<LFileName<<", "<<RFileName<<"\n";

	if(Limg.empty())
		return false;

	if(Rimg.empty())
		return false;

	Limg.copyTo(leftImg);
	Rimg.copyTo(rightImg);

	sequenceCounter++;

	return true;
}

input::~input() {
	// TODO Auto-generated destructor stub
}

