/*
 * pclStereo.cpp
 *
 *  Created on: Oct 3, 2016
 *      Author: xwong
 */

#include "pclStereo.h"

pclStereo::pclStereo() {
	// TODO Auto-generated constructor stub

	  stereo.setMaxDisparity(128);
	  stereo.setXOffset(0);
	  stereo.setRadius(5);

	  stereo.setRatioFilter(20);
	  stereo.setPeakFilter(0);

	  stereo.setLeftRightCheck(true);
	  stereo.setLeftRightCheckThreshold(1);

	  stereo.setPreProcessing(true);

}

void pclStereo::solveStereo(cv::Mat imgL, cv::Mat imgR,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr outPointCloudPtr, float u0, float v0, float f, float b) {


	pcl::PointCloud<pcl::RGB>::Ptr left_cloud (new pcl::PointCloud<pcl::RGB>);
	pcl::PointCloud<pcl::RGB>::Ptr right_cloud (new pcl::PointCloud<pcl::RGB>);

	cv2pcd.convert(imgL,left_cloud);
	cv2pcd.convert(imgR,right_cloud);

	stereo.compute(*left_cloud,  *right_cloud);

	stereo.medianFilter(4);

	stereo.getPointCloud(u0, v0, f, b, outPointCloudPtr, left_cloud);
}

pclStereo::~pclStereo() {
	// TODO Auto-generated destructor stub
}

