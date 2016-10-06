/*
 * pclStereo.h
 *
 *  Created on: Oct 3, 2016
 *      Author: xwong
 */

#ifndef PCLSTEREO_H_
#define PCLSTEREO_H_

#include <pcl/stereo/stereo_matching.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/image_viewer.h>

#include <pcl/io/pcd_io.h>

#include "img2pcd.h"

class pclStereo {
private:

	  pcl::BlockBasedStereoMatching stereo;

	  img2pcd cv2pcd;


public:
	pclStereo();

	void solveStereo(cv::Mat imgL, cv::Mat imgR, pcl::PointCloud<pcl::PointXYZRGB>::Ptr outPointCloudPtr, float u0, float v0, float f, float b);

	virtual ~pclStereo();
};

#endif /* PCLSTEREO_H_ */
