/*
 * input.h
 *
 *  Created on: Sep 9, 2016
 *      Author: xwong
 */

/*
 * Read image / video input and split into Left / Right image sequence, and pass to
 */


#ifndef INPUT_H_
#define INPUT_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

class input {
private:

	std::string fileExtension;

	int sequenceCounter;

public:
	/*
	 * standard constructor with fileExtension = pgm, sequenceCounter =0
	 */
	input();

	/*
	 * Constructor with file extension setting
	 * @param fileExt input images file extension
	 * @param initial file counter, default = 0
	 */
	input(std::string fileExt, int initCounter=0)
	{
		fileExtension = fileExt;
		sequenceCounter = initCounter;
	}

	/*
	 * @param leftImg  left image
	 * @param rightImg right image
	 * @return false when image sequence end
	 */
	bool update(cv::Mat& leftImg, cv::Mat& rightImg);

	/*
	 * standard destructor
	 */
	virtual ~input();
};

#endif /* INPUT_H_ */
