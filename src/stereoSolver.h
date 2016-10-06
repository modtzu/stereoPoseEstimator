/*
 * stereoSolver.h
 *
 *  Created on: Sep 9, 2016
 *      Author: xwong
 *
 *     Solving 3D point cloud from stereo pair
 */

/*
 * Compute disparity, and 3D map
 * current version support stereo camera only
 *
 * future will need to added multiple base line stereo + stereo with orientation different
 */

#ifndef STEREOSOLVER_H_
#define STEREOSOLVER_H_

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <armadillo>

#include <vector>
#include <iostream>

class stereoSolver {
private:

	/// stereo camera setting
	double u0; // image principle point x
	double v0; // image principle point y
	double f;  // focus length
	double B;  // baseline distance

public:
	stereoSolver();

	stereoSolver(double u0, double v0, double f, double B)
	{
		this->u0 = u0;
		this->v0 = v0;
		this->f = f;
		this->B = B;
	}

//	bool computeDepth(std::vector<cv::Point2f>* ptrFt0, std::vector<cv::Point2f>* ptrFt1, std::vector<cv::Vec3f>& vctPt);

	bool computeDepth(std::vector<cv::KeyPoint>* ptrFt0, std::vector<cv::KeyPoint>* ptrFt1, std::vector<cv::Vec3f>& vctPt);

	virtual ~stereoSolver();
};

#endif /* STEREOSOLVER_H_ */
