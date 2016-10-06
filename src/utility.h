/*
 * utility.h
 *
 *  Created on: Jul 2, 2015
 *      Author: xwong
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#include "opencv2/opencv.hpp"
#include "armadillo"

class utility {
public:
	utility();

	arma::vec2 objLocToImgLoc(arma::vec3 objLoc, arma::mat camRot, arma::vec3 camTrans, arma::mat camIntrin);

	arma::vec3 cameraRayDir(int px, int py, arma::mat camIntrin);

	arma::vec3 rayRayIntersection(arma::vec3 P1, arma::vec3 D1, arma::vec3 P2, arma::vec3 D2 );

	cv::Vec3f convolutedIntensity(cv::Point2i pLoc, double sigma, cv::Mat img);

	arma::vec3 getTanFromNormal(arma::vec3 N,  cv::Vec2f dir);

	void showHistogram(cv::Mat img);

	void histRGB(cv::Mat img, int low, int up);

	void histGray(cv::Mat img, int low, int up);

	double FastNoiseVariance(cv::Mat img);

	double FastNoiseVariance(cv::Mat img, cv::Mat featureMap);

	cv::Mat FastNoiseVarianceMap(cv::Mat img, cv::Mat featureMap);

	cv::Mat gConvoluteEdgeDetector(cv::Mat img);

	arma::vec4 RtoQ(arma::mat R);

	arma::mat QtoR(arma::vec4 Q);

	double getLocalDynRng(cv::Mat img);

	cv::RotatedRect plotCovmat(double chisquare_val, cv::Point2f mean, arma::mat covmat);

	virtual ~utility();
};

#endif /* UTILITY_H_ */
