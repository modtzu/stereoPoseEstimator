/*
 * imgFtRelated.h
 *
 *  Created on: Jun 23, 2015
 *      Author: xwong
 */

/*
 *description : function related to image feature extraction and tracking.
 */

#ifndef SRC_IMGFTRELATED_H_
#define SRC_IMGFTRELATED_H_

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"


#include <vector>
#include <deque>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#include "armadillo"

using namespace std;

class imgFtRelated {

private:
	cv::SiftFeatureDetector* ptrFeatureDetector;

	std::ofstream ftWriter;

public:

//	int maxNumFt = 100, int octaveLvl = 3, int contrastTh = 0.1, int edgeTh = 20, int sigma = 2.6

	/*
	 *@note constructor with SIFT feature extraction setup
	 *@input maxNumFt maximum number of feature, default = 100
	 *@input octaveLvl SIFT octave level, default = 3
	 *@input constratTh SIFT contrast threshold
	 *@input edgeTh SIFT edge threshold
	 *@input sigma SIFT sigma vaue
	 */
	imgFtRelated(int maxNumFt = 100, int octaveLvl = 3, int contrastTh = 0.0001, int edgeTh = 10, int sigma =1.6)
	{
		ptrFeatureDetector = new cv::SiftFeatureDetector(maxNumFt,octaveLvl,contrastTh, edgeTh, sigma);
	}

	/*
	 *@note extract feature point
	 *@input img input image
	 *@input Ft output feature track
	 *@input descriptor output SIFT descriptor
	 *@input flg feature extractor selection, 0 = SIFT, 1 = good feature to track
	 */
	void getFeature(cv::Mat img, vector<cv::KeyPoint> &Ft, cv::Mat& descriptor, int flg = 0);

	/*
	 *@note extract feature with good feature to track method
	 *@input img input image
	 *@input Ft output vector of feature
	 */
	void getFeatureGdtt(cv::Mat img, vector<cv::KeyPoint>& Ft);

	/*
	 *@note get SIFT descriptor for non SIFT feature ectracted feature points
	 *@input img input image
	 *@input kFt vector of feature points
	 *@input descriptor output feature descriptor
	 */
	void getDescriptorForPt(cv::Mat img,vector<cv::KeyPoint> kFt, cv::Mat& descriptor);

	/*
	 *@note match feaute with SIFT descriptot matching
	 *@input img1 image 1
	 *@input img2 image 2
	 *@input kFT1 feature track 1
	 *@input kFT2 feature track 2
	 *@input Ft0 feature track 1 in Point2f format /// note unnecessary, remove in future
	 *@input Ft1 feature trac 2 in Point2f format
	 *@input descriptor descriptor for feature track 1
	 *@input descriptor descriptor for feature track 2
	 *@output false if not match found
	 */
	bool matchFeature(cv::Mat img1, cv::Mat img2, vector<cv::KeyPoint>& kFt1, vector<cv::KeyPoint>& kFt2,
			vector<cv::Point2f> &Ft0,  vector<cv::Point2f> &Ft1, cv::Mat descriptor1, cv::Mat descriptor2);

	/*
	 *@note track feature point with opencv KLT function
	 *@input img0 image 1
	 *@input img1 image 2
	 *@input Ft0 feature points 1 /// update after track, remove points that is failed to track
	 *@input Ft1 tracked feature points 2
	 *@input removed ID vector of index indicating failed to track feature
	 *@input kltWinSize windows size for KLT tracker
	 */
	void trackFeature(cv::Mat img0, cv::Mat img1, vector<cv::KeyPoint> &Ft0,  vector<cv::KeyPoint> &Ft1, vector<int>& removedID, cv::Size KltWinSize = cv::Size(50,50));

	/*
	 *@note save key points into .txt file
	 *@input Ft feature points
	 *@input ID index
	 */
	void saveKeyPoint(vector<cv::KeyPoint> Ft, int ID);

	/*
	 *@home made KLT tracker 1, solve using full matrix form
	 *@input img0 image 0
	 *@input img1 image 1
	 *@input Ft0 origin feature point
	 *@input winSize KLT windows size
	 *@output tracked feature location
	 */
	cv::Point2f hmKLT(cv::Mat img0, cv::Mat img1, cv::Point2f Ft0, cv::Size winSize);

	/*
	 *@home made KLT tracker 2, solve using summation form yield better efficiency than hmKLT
	 *@input img0 image 0
	 *@input img1 image 1
	 *@input Ft0 origin feature point
	 *@input winSize KLT windows size
	 *@output tracked feature location
	 */
	cv::Point2f hmCvKLT(cv::Mat img0, cv::Mat img1, cv::Point2f Ft0, cv::Size winSize);


	/*
	 *@home made KLT tracker 3, solve using summation form and opencv wrap function yield identical result with openCV KLT tracker function
	 *@input img0 image 0
	 *@input img1 image 1
	 *@input Ft0 origin feature point
	 *@input winSize KLT windows size
	 *@output tracked feature location
	 */
	cv::Point2f hmCvWrapKLT(cv::Mat img0, cv::Mat img1, cv::Point2f Ft0, cv::Size winSize);

	/*
	 *@note test if KLT windows size is within the image
	 *@input imgSize image size
	 *@input C1 window centroid
	 *@input winSize window size
	 */
	bool testWinSize(cv::Size imgSize, cv::Point2f& C1, cv::Size& winSize);

	/*
	 *@ home made affine klt tracker, not working...
	 */
	cv::Point2f hmAffineKLT(cv::Mat img0, cv::Mat img1, cv::Point2f Ft0, cv::Size winSize);

	/*
	 *@note home made pyramid KLT tracker
	 */
	cv::Point2f hmPryKLT(cv::Mat img0, cv::Mat img1, cv::Point2f Ft0, cv::Size winSize, int pryLvl);

	/*
	 *@note pyramid constructor for hmPryKLT
	 */
	cv::Mat pyramidConstructor(cv::Mat img0);

	/*
	 *@note solve kernel matrix for 1st order derivative along x direction
	 *@input kernelSize kernel size
	 *@input sigma sigma
	 *@output kernel Matrix
	 */
	cv::Mat dGdXkernel(cv::Size kernelSize, double sigma);

   /*
	*@note solve kernel matrix for 1st order derivative along y direction
	*@input kernelSize kernel size
	*@input sigma sigma
	*@output kernel Matrix
	*/
	cv::Mat dGdYkernel(cv::Size kernelSize, double sigma);

	/*
	 *@note check feature point with harris corner detector to make sure it is a point
	 *@input point point
	 *@input img image
	 *@return true if it is a point
	 */
	bool harrisCornerCheck(cv::Point2f point, cv::Mat img);

	/*
	 *@note matching feature along epipolar line
	 *@note assume epipolar line is parallel to horizontal axis
	 */
	bool stereoMatch(cv::Mat img0, cv::Mat img1, cv::KeyPoint Ft0, cv::KeyPoint& Ft1 ,cv::Size winSize);

	virtual ~imgFtRelated();
};

#endif /* SRC_IMGFTRELATED_H_ */
