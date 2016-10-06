/*
 * img2pcd.h
 *
 *  Created on: Sep 8, 2016
 *      Author: xwong
 */

/*
 * Convert general image input to pcd format
 */


#ifndef IMG2PCD_H_
#define IMG2PCD_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>

class img2pcd {

public:
	img2pcd();

	/*
	 * Convert image in Mat format to Point Cloud format
	 * @param inputImg input image in openCV Mat format
	 * @param pcd pointer to point cloud storage
	 * @return false if failed
	 */
	bool convert(cv::Mat inputImg, pcl::PointCloud<pcl::RGB>::Ptr pcd);

	virtual ~img2pcd();
};

#endif /* IMG2PCD_H_ */
