/*
 * pcVIsual.cpp
 *
 *  Created on: Sep 12, 2016
 *      Author: xwong
 */

#include "pcVIsual.h"

pcVIsual::pcVIsual() {
	// TODO Auto-generated constructor stub

}

bool pcVIsual::visualize(std::string winName, cv::Mat inputImg, cv::Mat refImg) {

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZRGB>);

	for(int yi = 0; yi < inputImg.rows; yi++)
		for(int xi = 0; xi < inputImg.cols; xi++)
			{
				pcl::PointXYZRGB point;

				/// get image format
				switch(inputImg.channels())
				{
				case 1: /// single channel mat

					point.x = xi;
					point.y = yi;
					point.z = inputImg.at<float>(yi,xi);

					break;

				case 3: /// mat of vec3f

					point.x = inputImg.at<cv::Vec3f>(yi,xi)[0];
					point.y = inputImg.at<cv::Vec3f>(yi,xi)[1];
					point.z = inputImg.at<cv::Vec3f>(yi,xi)[2];

					break;
				}

				/// get image format
				switch(refImg.channels())
				{
				case 1: /// single channel mat

					point.r = refImg.at<uchar>(yi,xi);
					point.g = refImg.at<uchar>(yi,xi);
					point.b = refImg.at<uchar>(yi,xi);

					break;

				case 3: /// mat of vec3f

					point.r = refImg.at<cv::Vec3b>(yi,xi)[0];
					point.g = refImg.at<cv::Vec3b>(yi,xi)[1];
					point.b = refImg.at<cv::Vec3b>(yi,xi)[2];

					break;
				}



				pcd->push_back(point);

			}

		return visualize(winName, pcd);

}

bool pcVIsual::visualize(std::string winName, arma::mat pcCV) {

}

bool pcVIsual::visualize(std::string winName,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcCV) {

	  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	  viewer->setBackgroundColor (0, 0, 0);

	  //viewer->addPointCloud<pcl::PointXYZRGB> (out_cloud, "stereo");

//	  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> intensity(out_cloud);
//	  viewer->addPointCloud<pcl::PointXYZRGB> (out_cloud, intensity, "stereo");

	  viewer->addPointCloud<pcl::PointXYZRGB> (pcCV,"stereo");

	  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "stereo");
	  viewer->initCameraParameters ();
	  //viewer->spin();
	  while (!viewer->wasStopped ())
	  {
	    viewer->spinOnce (100);
	    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	  }

	  return true;


//	pcl::visualization::CloudViewer viewer(winName);
//	viewer.showCloud(pcCV);
//
//	while (!viewer.wasStopped ())
//	  {
//	  }
//
//	return true;

//	pcl::visualization::PCLVisualizer viewer (winName);
//	viewer.addPointCloud(pcCV,"original_cloud");
//
//	viewer.addCoordinateSystem (1.0, "cloud", 0);
//	viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
//	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
//
//	  while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
//	    viewer.spinOnce ();
//	  }

	return true;
}

pcVIsual::~pcVIsual() {
	// TODO Auto-generated destructor stub
}

bool pcVIsual::visualize(std::string winName, std::vector<cv::Vec3f> vctP) {


	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZRGB>);

	for(int i = 0; i < vctP.size(); i++)
		{
			pcl::PointXYZRGB point;

			point.x = vctP[i][0];
			point.y = vctP[i][1];
			point.z = vctP[i][2];

			point.r = 255;
			point.g = 255;
			point.b = 255;

			pcd->push_back(point);
		}


	pcl::visualization::CloudViewer viewer(winName);
	viewer.showCloud(pcd);

	while (!viewer.wasStopped ())
	  {
		usleep(1000);
	  }

	return true;

}


void pcVIsual::addPt2Window(std::string winName, std::vector<cv::Vec3f> vctP, std::string ptName, Eigen::Matrix4f transformMat) {

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZRGB>);

	for(int i = 0; i < vctP.size(); i++)
		{
			pcl::PointXYZRGB point;

			point.x = vctP[i][0];
			point.y = vctP[i][1];
			point.z = vctP[i][2];

			point.r = 255;
			point.g = 255;
			point.b = 255;

			pcd->push_back(point);
		}

	pcl::transformPointCloud(*pcd, *pcd, transformMat);

	 auto iter = std::find(vctWinName.begin(),vctWinName.end(),winName);

	 if(iter == vctWinName.end())
	 {
		 vctWinName.push_back(winName);
		 boost::shared_ptr<pcl::visualization::PCLVisualizer> wis(new pcl::visualization::PCLVisualizer(winName));
		 vctPclVisualizerPtr.push_back(wis);

		 int s = vctPclVisualizerPtr.size()-1;

		 vctPclVisualizerPtr[s]->setBackgroundColor(0,0,0);
		 vctPclVisualizerPtr[s]->addPointCloud<pcl::PointXYZRGB> (pcd,ptName);

	 }
	 else
	 {
		 int it = iter - vctWinName.begin();
		 vctPclVisualizerPtr[it]->addPointCloud<pcl::PointXYZRGB> (pcd,ptName);
	 }


}

void pcVIsual::showWindow(std::string winName) {

	auto iter = std::find(vctWinName.begin(),vctWinName.end(),winName);

	 if(iter != vctWinName.end())
	 {
		 int it = iter - vctWinName.begin();

		 vctPclVisualizerPtr[it]->setBackgroundColor(0,0,0);
		 vctPclVisualizerPtr[it]->initCameraParameters();

		 while (!vctPclVisualizerPtr[it]->wasStopped ())
		  {
			vctPclVisualizerPtr[it]->spinOnce (100);
		    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
		  }

	 }


}

void pcVIsual::addPt2Window(std::string winName,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclPtr, std::string ptName,
		Eigen::Matrix4f transformMat) {

	pcl::transformPointCloud(*pclPtr, *pclPtr, transformMat);

	 auto iter = std::find(vctWinName.begin(),vctWinName.end(),winName);

	 if(iter == vctWinName.end())
	 {
		 vctWinName.push_back(winName);
		 boost::shared_ptr<pcl::visualization::PCLVisualizer> wis(new pcl::visualization::PCLVisualizer(winName));
		 vctPclVisualizerPtr.push_back(wis);

		 int s = vctPclVisualizerPtr.size()-1;

		 vctPclVisualizerPtr[s]->setBackgroundColor(0,0,0);
		 vctPclVisualizerPtr[s]->addPointCloud<pcl::PointXYZRGB> (pclPtr,ptName);

	 }
	 else
	 {
		 int it = iter - vctWinName.begin();
		 vctPclVisualizerPtr[it]->addPointCloud<pcl::PointXYZRGB> (pclPtr,ptName);
	 }

}
