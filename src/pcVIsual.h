/*
 * pcVIsual.h
 *
 *  Created on: Sep 12, 2016
 *      Author: xwong
 */

#ifndef PCVISUAL_H_
#define PCVISUAL_H_

#include <pcl/stereo/stereo_matching.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <armadillo>

class pcVIsual {
private:

	std::vector<std::string> vctWinName;

	std::vector< std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > vctPointCloud;

	std::vector< boost::shared_ptr<pcl::visualization::PCLVisualizer> > vctPclVisualizerPtr;

public:
	pcVIsual();

	bool visualize(std::string winName, cv::Mat pcCV, cv::Mat refImg);

	bool visualize(std::string winName, arma::mat pcCV);

	bool visualize(std::string winName, std::vector<cv::Vec3f> vctP);

	bool visualize(std::string winName, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcCV);

	void addPt2Window(std::string winName, std::vector<cv::Vec3f> vctP, std::string ptName, Eigen::Matrix4f transformMat = Eigen::Matrix4f::Identity());

	void addPt2Window(std::string winName, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclPtr, std::string ptName, Eigen::Matrix4f transformMat = Eigen::Matrix4f::Identity());

	void showWindow(std::string winName);

	virtual ~pcVIsual();
};

#endif /* PCVISUAL_H_ */
