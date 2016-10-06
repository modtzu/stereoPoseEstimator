/*
 * KltWithCov.h
 *
 *  Created on: Feb 17, 2016
 *      Author: xwong
 */

#ifndef KLTWITHCOV_H_
#define KLTWITHCOV_H_

#include "imgFtRelated.h"
#include "kltUncertainty.h"

#include <deque>

struct stKltCov
{
	int imgRng[2];
	std::vector< std::vector<cv::KeyPoint> > vctVctFtTrack; /// vctVctFtTrack[Frame][Ft]
	std::vector< std::vector<arma::mat> > vctVctFtCov;
};

class KltWithCov {
private:

	imgFtRelated FTT;
	kltUncertainty KLTU;

	std::vector<stKltCov> vctStKltCovOut;
	std::vector<cv::Mat> vctImg;

	cv::Size KltWinSize;

public:
	KltWithCov();

	void track(std::vector<cv::Mat> vctImg, std::vector<stKltCov>& outputStKltCov, cv::Size KltWinSize, int feaMode = 0);

	virtual ~KltWithCov();

private:

	void trackLoop(int feaMode);

	void InitFeature(int imgID, std::deque<cv::KeyPoint>& initFt,std::deque<arma::mat>& initCov,int feaMode);

	void trackFeature(int imgID0, int imgID1, std::deque<cv::KeyPoint> initFt, std::deque<cv::KeyPoint>& trackFt, std::vector<int>& ejectID);

	void updateCov(int imgID0, int imgID1, std::deque<cv::KeyPoint> initFt, std::deque<cv::KeyPoint> trackFt,
			std::deque<arma::mat> initCov, std::deque<arma::mat>& trackCov, std::vector<int>& ejectID);

	void ejectFt(std::deque<cv::KeyPoint>& Ft, std::deque<arma::mat> &Cov, std::vector<int> ejectID);

	void ejectTrack(std::deque< std::vector<cv::KeyPoint> >& vctVctFtTrack, std::deque< std::vector<arma::mat> >& vctVctFtCov, std::vector<int> ejectID);

};

#endif /* KLTWITHCOV_H_ */
