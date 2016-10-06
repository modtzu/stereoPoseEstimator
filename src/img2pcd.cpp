/*
 * img2pcd.cpp
 *
 *  Created on: Sep 8, 2016
 *      Author: xwong
 */

#include "img2pcd.h"

img2pcd::img2pcd() {
	// TODO Auto-generated constructor stub

}

bool img2pcd::convert(cv::Mat inputImg, pcl::PointCloud<pcl::RGB>::Ptr pcd) {

	pcd->clear();

for(int yi = 0; yi < inputImg.rows; yi++)
	for(int xi = 0; xi < inputImg.cols; xi++)
		{
			pcl::RGB point;

			/// get image format
			switch(inputImg.channels())
			{
			case 1: /// Gray scale image


				point.r = inputImg.at<uchar>(yi,xi);
				point.g = inputImg.at<uchar>(yi,xi);
				point.b = inputImg.at<uchar>(yi,xi);
				point.a = 1;

				break;

			case 3: /// RGB image

				point.r = inputImg.at<cv::Vec3b>(yi,xi)[0];
				point.g = inputImg.at<cv::Vec3b>(yi,xi)[1];
				point.b = inputImg.at<cv::Vec3b>(yi,xi)[2];
				point.a = 1;

				break;
			}

			pcd->push_back(point);

		}

	pcd->width = inputImg.cols;
	pcd->height = inputImg.rows;

	return true;
}

img2pcd::~img2pcd() {
	// TODO Auto-generated destructor stub
}

