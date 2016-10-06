/*
 * KltWithCov.cpp
 *
 *  Created on: Feb 17, 2016
 *      Author: xwong
 */

#include "KltWithCov.h"

KltWithCov::KltWithCov() {
	// TODO Auto-generated constructor stub

}

void KltWithCov::track(std::vector<cv::Mat> vctImg,
		std::vector<stKltCov>& outputStKltCov,cv::Size KltWinSize, int feaMode) {
	this->vctImg = vctImg;

	this->KltWinSize = KltWinSize;

	trackLoop(feaMode);

	outputStKltCov = this->vctStKltCovOut;
}

KltWithCov::~KltWithCov() {
	// TODO Auto-generated destructor stub
}

void KltWithCov::trackLoop(int feaMode) {
	std::deque<cv::KeyPoint> preFt;
	std::deque<arma::mat> preCov;

	std::deque< std::vector<cv::KeyPoint> > vctVctFtTrack; /// vctVctFtTrack[Frame][Ft]
	std::deque< std::vector<arma::mat> > vctVctFtCov;

	stKltCov stTemp;

	stTemp.imgRng[0] = 0;

	InitFeature(0,preFt,preCov,feaMode);

	vctVctFtTrack.push_back(std::vector<cv::KeyPoint>(preFt.begin(),preFt.end()));
	vctVctFtCov.push_back(std::vector<arma::mat>(preCov.begin(),preCov.end()));

	unsigned int i = 1;

	for(i = 1; i < vctImg.size(); i++)
	{
		std::deque<cv::KeyPoint> trackFt;
		std::deque<arma::mat> trackCov;

		std::vector<int> ejectID;

		if(preFt.empty())
			break;

		/// track feature from img[i-1] to img[i]
		trackFeature(i-1,i,preFt,trackFt,ejectID);

		std::cout<<i<<" "<<preFt.size()<<" "<<trackFt.size()<<"\n";

		int nFr = 10;//vctImg.size();
		int nfe = 20;

		/// Some Points are failed to track by KLT, eject feature points if frame < 3
		if(!ejectID.empty() && vctVctFtTrack.size() < nFr)
		{
			ejectFt(preFt,preCov,ejectID);
//			ejectFt(trackFt,trackCov,ejectID);
			ejectTrack(vctVctFtTrack,vctVctFtCov,ejectID);

			ejectID.clear();
		}

		if(!ejectID.empty() || trackFt.size()<nfe)
		{
			/// reset
			stTemp.imgRng[1] = i-1;
			stTemp.vctVctFtCov = (std::vector< std::vector<arma::mat> >(vctVctFtCov.begin(),vctVctFtCov.end()));
			stTemp.vctVctFtTrack = (std::vector< std::vector<cv::KeyPoint> >(vctVctFtTrack.begin(),vctVctFtTrack.end()));
			vctStKltCovOut.push_back(stTemp);

			vctVctFtTrack.clear();
			vctVctFtCov.clear();

			i --;
			stTemp.imgRng[0] = i;

			InitFeature(i,preFt,preCov,feaMode);

			vctVctFtTrack.push_back(std::vector<cv::KeyPoint>(preFt.begin(),preFt.end()));
			vctVctFtCov.push_back(std::vector<arma::mat>(preCov.begin(),preCov.end()));
			ejectID.clear();

			continue;
		}


		/// updateCovariance
		updateCov(i-1,i,preFt,trackFt,preCov,trackCov,ejectID);

		/// eject points with invalid covariance
		/// Some Points are failed to track by KLT, eject feature points if frame < 3
		if(!ejectID.empty() && vctVctFtTrack.size() < nFr)
		{
			ejectFt(preFt,preCov,ejectID);
			ejectFt(trackFt,trackCov,ejectID);
			ejectTrack(vctVctFtTrack,vctVctFtCov,ejectID);
			ejectID.clear();
		}

		if(!ejectID.empty() || trackFt.size()<nfe)
		{
			/// reset
			stTemp.imgRng[1] = i-1;
			stTemp.vctVctFtCov = (std::vector< std::vector<arma::mat> >(vctVctFtCov.begin(),vctVctFtCov.end()));
			stTemp.vctVctFtTrack = (std::vector< std::vector<cv::KeyPoint> >(vctVctFtTrack.begin(),vctVctFtTrack.end()));
			vctStKltCovOut.push_back(stTemp);

			vctVctFtTrack.clear();
			vctVctFtCov.clear();

			i --;
			stTemp.imgRng[0] = i;

			InitFeature(i,preFt,preCov,feaMode);

			vctVctFtTrack.push_back(std::vector<cv::KeyPoint>(preFt.begin(),preFt.end()));
			vctVctFtCov.push_back(std::vector<arma::mat>(preCov.begin(),preCov.end()));
			ejectID.clear();

			continue;
		}


			/// add tracked data to buffer
			vctVctFtTrack.push_back(std::vector<cv::KeyPoint>(trackFt.begin(),trackFt.end()));
			vctVctFtCov.push_back(std::vector<arma::mat>(trackCov.begin(),trackCov.end()));

			preFt = trackFt;
			preCov = trackCov;

	}

	/// End of loop and Factorization need at least 3 frames
//	if(vctVctFtTrack.size()>3)
	{
		/// save storage buffer
		stTemp.imgRng[1] = stTemp.imgRng[0] + vctVctFtCov.size() - 1;
		stTemp.vctVctFtCov = (std::vector< std::vector<arma::mat> >(vctVctFtCov.begin(),vctVctFtCov.end()));
		stTemp.vctVctFtTrack = (std::vector< std::vector<cv::KeyPoint> >(vctVctFtTrack.begin(),vctVctFtTrack.end()));
		vctStKltCovOut.push_back(stTemp);
	}

}

void KltWithCov::InitFeature(int imgID, std::deque<cv::KeyPoint>& initFt,
		std::deque<arma::mat>& initCov, int feaMode) {

	cv::Mat descriptor;

	std::vector<cv::KeyPoint> vctInitFt;

	FTT.getFeature(vctImg[imgID],vctInitFt,descriptor,feaMode);

	initFt.clear();
	initFt = std::deque<cv::KeyPoint>(vctInitFt.begin(),vctInitFt.end());

	initCov.clear();

	for(int i=0 ; i < initFt.size(); i++)
	{
		arma::mat estInitCov(2,2);
//		estInitCov.zeros();
		estInitCov.eye();

		initCov.push_back(estInitCov);
	}

}

void KltWithCov::trackFeature(int imgID0, int imgID1, std::deque<cv::KeyPoint> initFt,
		std::deque<cv::KeyPoint>& trackFt, std::vector<int>& ejectID) {

	std::vector<cv::KeyPoint> vctPt0 = std::vector<cv::KeyPoint>(initFt.begin(),initFt.end()) ;
	std::vector<cv::KeyPoint> vctPt1;

	FTT.trackFeature(vctImg[imgID0],vctImg[imgID1],vctPt0,vctPt1,ejectID, KltWinSize);

	trackFt = std::deque<cv::KeyPoint>(vctPt1.begin(),vctPt1.end());
}

void KltWithCov::updateCov(int imgID0, int imgID1, std::deque<cv::KeyPoint> initFt,
		std::deque<cv::KeyPoint> trackFt, std::deque<arma::mat> initCov,
		std::deque<arma::mat>& trackCov, std::vector<int>& ejectID) {


	trackCov = initCov;

	ejectID.clear();

	for(int i= 0; i < initFt.size(); i++)
	{
		arma::mat tCov = KLTU.kltCovLS(vctImg[imgID0],
											 vctImg[imgID1],
													KltWinSize,
													initFt[i],
													trackFt[i],
													initCov[i],
													cv::Size(16,16),
													1.6);
//		tCov.eye(2,2);

		if(tCov(0,0) <0 && tCov(1,1) <0)
			{
				ejectID.push_back(i);
			}
		else
		{
			trackCov[i] = tCov;
		}
	}

}

void KltWithCov::ejectFt(std::deque<cv::KeyPoint>& Ft,
		std::deque<arma::mat>& Cov, std::vector<int> ejectID) {

	int eraseCount = 0;

	for(unsigned int i =0; i < ejectID.size(); i++)
	{
		if(Ft.size() > ejectID[i] - eraseCount)
			Ft.erase(Ft.begin() + ejectID[i] - eraseCount);

		if(Cov.size() > ejectID[i] - eraseCount)
			Cov.erase(Cov.begin() + ejectID[i] - eraseCount);

		eraseCount ++;
	}

}

void KltWithCov::ejectTrack(
		std::deque<std::vector<cv::KeyPoint> >& vctVctFtTrack,
		std::deque<std::vector<arma::mat> >& vctVctFtCov,
		std::vector<int> ejectID) {

	int eraseCount = 0;

	for(unsigned int i =0; i < ejectID.size(); i++)
	{

		for(unsigned int k =0; k < vctVctFtTrack.size(); k++)
		{
			if(vctVctFtTrack[k].size() > ejectID[i] - eraseCount)
				vctVctFtTrack[k].erase(vctVctFtTrack[k].begin() + ejectID[i] - eraseCount);

			if(vctVctFtCov[k].size() > ejectID[i] - eraseCount)
				vctVctFtCov[k].erase(vctVctFtCov[k].begin() + ejectID[i] - eraseCount);
		}

		eraseCount ++;
	}

}
