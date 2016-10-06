/*
 * main.cpp
 *
 *  Created on: Sep 8, 2016
 *      Author: xwong
 *
 * 	estimate relative pose from stereo observation and create a global map
 *
 */

#include "input.h"
#include "imgFtRelated.h"
#include "pcVIsual.h"
#include "stereoSolver.h"
#include "ppTransEst.h"
#include "pclStereo.h"

#include "kltUncertainty.h"
#include "utility.h"


ppTransEst ppTrans;
pcVIsual vi;
kltUncertainty KLTU;
utility UT;
pclStereo pclStr;

/// argument parser
void argumentParser(int argc, char** argv, std::string& fileExt, std::string& fileName,
		std::string& camCalibfile,std::string& optionFile, int& initCounter)
{

	for(int i =0; i < argc; i++)
	{
		if(strcmp("-e",argv[i])== 0)
		{
			fileExt = std::string(argv[i+1]);
		}
		else
		if(strcmp("-f",argv[i])== 0)
		{
			fileName = std::string(argv[i+1]);
		}
		else
		if(strcmp("-c",argv[i])== 0)
		{
			camCalibfile = std::string(argv[i+1]);
		}
		else
		if(strcmp("-o",argv[i])== 0)
		{
			optionFile = std::string(argv[i+1]);
		}
		else
		if(strcmp("-i",argv[i])== 0)
		{
			initCounter = atoi(argv[i+1]);
		}
	}
}


int main(int argc, char** argv)
{
	std::string fileExt, fileName, camCalibFile, optionFile;
	int initCounter = 0;

	/// parse argument
	argumentParser(argc, argv,fileExt,fileName, camCalibFile,optionFile,initCounter);

	/// setup image input reader
	input imgIn(fileExt,initCounter);

	/// Image at t = k-1
	cv::Mat imgR0,imgL0;

	/// get image at t = 0
	if(!imgIn.update(imgL0,imgR0))
	{
		std::cout<<"Error reading image input\n";
		return false;
	}

	/// feature storage
	std::vector<cv::KeyPoint> vctFt0L, vctFt0R;
	std::vector<cv::Point2f> Pt0, Pt1;
	std::vector<int> vctReID;
	cv::Mat des0,des1;

	/// Extract feature point from L0, and track to R0
	imgFtRelated feature(100, 3, 0.0001,10,1.6);
	feature.getFeature(imgL0,vctFt0L,des0,0);

	/// Establish key points correspondence in initial stereo pair with KLT tracker
	feature.trackFeature(imgL0,imgR0,vctFt0L,vctFt0R,vctReID,cv::Size(40,40));

	/// get SIFT descriptor for tracked feature
	feature.getDescriptorForPt(imgL0,vctFt0L,des0);
	feature.getDescriptorForPt(imgR0,vctFt0R,des1);

	/// descriptor matching for remove false track / outlier
	feature.matchFeature(imgL0,imgR0,vctFt0L,vctFt0R,Pt0,Pt1,des0,des1);

	cv::Mat kf, kfR;
	cv::drawKeypoints(imgL0,vctFt0L,kf,cv::Scalar(0,255,255));
	cv::drawKeypoints(imgR0,vctFt0R,kfR,cv::Scalar(0,255,255));

	cv::imshow("kfR0",kfR);
	cv::imshow("kf0",kf);

	cv::waitKey(0);

	/*
	 * Stereo camera setting
	 */
	double u0 = 305.408;
	double v0 = 244.826;
	double f = 808.083;
	double b = 0.240272;

	/*
	 * Compute full 3D map
	 */
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloudPtr( new pcl::PointCloud<pcl::PointXYZRGB> );

	pclStr.solveStereo(imgL0,imgR0,pointCloudPtr,u0,v0,f,b);

	vi.addPt2Window("fullMap",pointCloudPtr,"0");


	/*
	 * keypoints 3D loc. in initial frame
	 */
	stereoSolver ste(u0,v0,f,b);
	std::vector<cv::Vec3f> vctPk,vctPkm1,vctPkm2;

	if(ste.computeDepth(&vctFt0L,&vctFt0R,vctPk))
	{
		vctPkm1 = vctPk;
		vctPkm2 = vctPk;
	}
	else
	{
		return false;
	}

	/*
	 *  relative pose estimation
	 */

	/// Image at t = k
	cv::Mat imgRt,imgLt;

	arma::mat Rt0, Tt0;
	Rt0.eye(3,3);
	Tt0.zeros(3,1);
	int c = 0;
	int Fr = 0;

	/// loop through input images sequence
	while(true)
	{
		/// get next image
		if(!imgIn.update(imgLt,imgRt))
		{
			std::cout<<"End of image sequence\n";
			break;
		}

		Fr ++;

		///Track feature from L0 - Lt, R0 - Rt
		std::vector<cv::KeyPoint> vctFt1L, vctFt1R;

		feature.trackFeature(imgL0,imgLt,vctFt0L,vctFt1L,vctReID,cv::Size(20,20));

		/// remove point that is lost track
		if(!vctReID.empty())
		{
			for(unsigned int i =0; i <vctReID.size();i++)
			{
				int idToRemove = vctReID[vctReID.size() - i - 1];

				vctPkm1.erase(vctPkm1.begin()+idToRemove);
				vctPkm2.erase(vctPkm2.begin()+idToRemove);
				vctPk.erase(vctPk.begin()+idToRemove);

				vctFt0R.erase(vctFt0R.begin()+idToRemove);
			}
		}


		/// Compute feature covariance and eject if uncertainty is large
		int CS = vctFt1L.size();
		for(int i=0; i<CS; i++)
		{
			int k = CS - i - 1;

			arma::mat initCov(2,2);
			initCov.eye();

			arma::mat tCov = KLTU.kltCovLS(imgL0,imgLt,cv::Size(20,20),vctFt0L[k],vctFt1L[k],initCov,
																cv::Size(5,5),
																1.6);

			arma::cx_vec eigVal;
			arma::cx_mat eigVec;


			arma::eig_gen(eigVal,eigVec,tCov);
			arma::mat eig = arma::real(eigVal);

			if(eig(0,0)>3 || eig(1,0) > 3 || eig(0,0)<= 0|| eig(1,0)<=0)
			{
				vctPkm1.erase(vctPkm1.begin()+k);
				vctPkm2.erase(vctPkm2.begin()+k);
				vctPk.erase(vctPk.begin()+k);

				vctFt0R.erase(vctFt0R.begin()+k);
				vctFt1L.erase(vctFt1L.begin()+k);
			}
		}

		/// track feature for right stereo image
		feature.trackFeature(imgR0,imgRt,vctFt0R,vctFt1R,vctReID,cv::Size(20,20));

		/// remove lost tracked features
		if(!vctReID.empty())
		{
			for(unsigned int i =0; i <vctReID.size();i++)
			{
				int idToRemove = vctReID[vctReID.size() - i - 1];
				vctPkm1.erase(vctPkm1.begin()+idToRemove);
				vctPkm2.erase(vctPkm2.begin()+idToRemove);
				vctFt1L.erase(vctFt1L.begin()+idToRemove);
			}
		}

		/// Compute feature covariance and eject if uncertainty is large
		CS = vctFt1R.size();
		for(int i=0; i<CS; i++)
		{
			int k = CS - i - 1;

			arma::mat initCov(2,2);
			initCov.eye();

			arma::mat tCov = KLTU.kltCovLS(imgR0,imgRt,cv::Size(20,20),vctFt0R[k],vctFt1R[k],initCov,
																cv::Size(5,5),
																1.6);

			arma::cx_vec eigVal;
			arma::cx_mat eigVec;

			arma::eig_gen(eigVal,eigVec,tCov);
			arma::mat eig = arma::real(eigVal);

			if(eig(0,0)>3 || eig(1,0) > 3 || eig(0,0)<= 0|| eig(1,0)<=0)
			{
				vctPkm1.erase(vctPkm1.begin()+k);
				vctPkm2.erase(vctPkm2.begin()+k);
				vctPk.erase(vctPk.begin()+k);
				vctFt1R.erase(vctFt1R.begin()+k);
				vctFt1L.erase(vctFt1L.begin()+k);
			}
		}

		std::vector<cv::Vec3f> vctPkp1;

		/// compute 3D point from current stereo images + estimate relative pose to previous frame
		if(ste.computeDepth(&vctFt1L,&vctFt1R,vctPkp1))
		{
			char ptName[256];
			sprintf(ptName,"%d",Fr);

			arma::mat R, T;
			std::vector<int> ejectID;

			/// solve relative pose between P1 and P0
//			ppTrans.solve(vctP1,vctP,R,T); /// solve with non linear least square

			/// solve with RANSAC remover
//			while(!ppTrans.solve(vctP,vctP1,R,T,ejectID))
//			{
//				CS = ejectID.size();
//
//				for(int i= 0; i < CS; i++)
//				{
//					int k = CS - i - 1;
//					vctP.erase(vctP.begin()+k);
//					vctFt1R.erase(vctFt1R.begin()+k);
//					vctFt1L.erase(vctFt1L.begin()+k);
//				}
//				ejectID.clear();
//			}


			/// solve with linear formulation + 2nd / 3rd order system model
			if(c < 1)
			{
				ppTrans.solveLinear(vctPk,vctPkp1,R,T,ejectID);
			}
			else if(c < 2)
				ppTrans.solveLinear2ndOrder(vctPkm1,vctPk,vctPkp1,R,T,ejectID); /// solve with linear least square 2nd order model
			else
				ppTrans.solveLinear3rdOrder(vctPkm2,vctPkm1,vctPk,vctPkp1,R,T,ejectID); /// solve with linear least square 3rd order model

		 /// transform estimated relative pose back into initial frame
			T = -R.t()*T;
			R = R.t();

			Tt0 = Tt0 + Rt0*T;
			Rt0 = R*Rt0;

		/// print camera relative position to initial frame
			std::cout<<vctPk.size()<<","<<Tt0.t();

			/// visualize key points in initial frame.
			Eigen::Matrix4f transformMat = Eigen::Matrix4f::Identity();

			for(int i=0 ; i < 3; i ++)
			{
				for(int j=0 ; j < 3; j ++)
				{
					transformMat(i,j) = Rt0(i,j);
				}
				transformMat(i,3) = Tt0(i,0);
			}

			if(c == 0 || c > 13)
			{
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloudPtr( new pcl::PointCloud<pcl::PointXYZRGB> );

				pclStr.solveStereo(imgL0,imgR0,pointCloudPtr,u0,v0,f,b);

				vi.addPt2Window("fullMap",pointCloudPtr,std::string(ptName),transformMat);

				if( c > 10)
					c = 3;
			}

		}

		/// visualize tracked feature points in each image
//		cv::Mat kf, kfR;
//		cv::drawKeypoints(imgLt,vctFt1L,kf,cv::Scalar(0,255,255));
//		cv::drawKeypoints(imgRt,vctFt1R,kfR,cv::Scalar(0,255,255));
//		cv::imshow("kfR",kfR);
//		cv::imshow("kf",kf);
//		cv::waitKey(0);

		/// update image
		imgRt.copyTo(imgR0);
		imgLt.copyTo(imgL0);

		/// if number of feature is least than 10, add new feature (current strategy is reset feature)
		if(vctPk.size()< 10)
		{
			vctFt0L.clear();
			vctFt0R.clear();

			feature.getFeature(imgL0,vctFt0L,des0,0);

			/// Track feature with KLT tracker
			feature.trackFeature(imgL0,imgR0,vctFt0L,vctFt0R,vctReID,cv::Size(40,40));

			/// get SIFT descriptor for tracked feature
			feature.getDescriptorForPt(imgL0,vctFt0L,des0);
			feature.getDescriptorForPt(imgR0,vctFt0R,des1);

			/// descriptor matching for remove false track / outlier
			feature.matchFeature(imgL0,imgR0,vctFt0L,vctFt0R,Pt0,Pt1,des0,des1);

			/// compute kep point location
			ste.computeDepth(&vctFt0L,&vctFt0R,vctPk);
			c = 0;
		}
		else
		{
			vctFt0L= vctFt1L;
			vctFt0R= vctFt1R;

			vctPkm2 = vctPkm1;
			vctPkm1 = vctPk;
			vctPk = vctPkp1;

			c++;
		}


	}

	vi.showWindow("fullMap");

	return 0;

	/// 3D points uncertainty
	/// relative pose uncertainty

	/// implement Kalman filter for subsequence state & covariance estimation

	/// Map / pose uncertainty

}
