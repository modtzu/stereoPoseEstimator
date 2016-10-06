/*
 * kltUncertainty.h
 *
 *  Created on: Aug 24, 2015
 *      Author: xwong
 */

#ifndef KLTUNCERTAINTY_H_
#define KLTUNCERTAINTY_H_

#include "imgFtRelated.h"

//#include "plotData.h"

class kltUncertainty {
private:
	imgFtRelated cvFt;
//	plotData pl;

	cv::Mat dGXk;
	cv::Mat dGYk;
	cv::Mat dGXXk;
	cv::Mat dGXYk;
	cv::Mat dGYYk;

public:
	kltUncertainty();

	double Gaussian(double x,double y,double sigma);

	cv::Mat LoG(cv::Mat img, cv::Size kernelSize, double sigma);

	double LoGkernel(cv::Point2f Pt, double sigma);

	cv::Mat dGdXkernel(cv::Size kernelSize, double sigma);

	cv::Mat dGdYkernel(cv::Size kernelSize, double sigma);

	cv::Mat ddGdXXkernel(cv::Size kernelSize, double sigma);

	cv::Mat ddGdYYkernel(cv::Size kernelSize, double sigma);

	cv::Mat ddGdXYkernel(cv::Size kernelSize, double sigma);

	arma::mat kltCovLS(cv::Mat img0, cv::Mat img1, cv::Size KLTwinSize,
			cv::KeyPoint Pt0, cv::KeyPoint Pt1, arma::mat covPre, cv::Size kernelSize=cv::Size(3,3), double sigma=1.6);

	arma::mat getSigmaUV(cv::Mat img0, cv::Mat img1, cv::Point2f Ft0,
			cv::Point2f Ft1,cv::Size KLTwinSize, cv::Mat G1X, cv::Mat G1Y);

	arma::mat getSigmaU(cv::Size KLTwinSize,cv::Mat G0X, cv::Mat G0Y);

	arma::mat dTdxyClose(cv::Mat G1X, cv::Mat G1Y,cv::Mat G0X, cv::Mat G0Y,
			cv::Mat H1X, cv::Mat H1Y, cv::Mat H1XY, cv::Mat sub0, cv::Mat sub1);

	arma::mat kAsTotalLS(arma::mat A, arma::mat Bx, arma::mat By, arma::mat D);

	virtual ~kltUncertainty();
};

#endif /* KLTUNCERTAINTY_H_ */
