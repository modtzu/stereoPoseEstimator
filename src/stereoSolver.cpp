/*
 * stereoSolver.cpp
 *
 *  Created on: Sep 9, 2016
 *      Author: xwong
 */

#include "stereoSolver.h"

stereoSolver::stereoSolver() {
	// TODO Auto-generated constructor stub
	 u0 = 0; // image principle point x
	 v0 = 0; // image principle point y
	 f = 0;  // focus length
	 B = 0;  // baseline distance
}

//
//bool stereoSolver::computeDepth(std::vector<cv::Point2f>* ptrFt0,
//		std::vector<cv::Point2f>* ptrFt1, std::vector<cv::Vec3f>& vctPt) {
//
//	if(u0 == 0 || v0 == 0)
//	{
//		std::cout<<"Error : Did not specify principle points\n";
//		return false;
//	}
//
//	if(f == 0 )
//	{
//		std::cout<<"Error : Did not specify camera focal length\n";
//		return false;
//	}
//
//	if(B == 0 )
//	{
//		std::cout<<"Error : Did not specify base line length\n";
//		return false;
//	}
//
//	if(ptrFt0->size()!=ptrFt1->size())
//	{
//		std::cout<<"Error : Ft0 size != Ft1 Size\n";
//		return false;
//	}
//
//	arma::mat dispMat(ptrFt0->size(),1);
//
//	vctPt.clear();
//
//	for(unsigned int i= 0; i < ptrFt0->size(); i++)
//	{
//		double D = sqrt(pow((*ptrFt1)[i].x-(*ptrFt0)[i].x,2)+pow((*ptrFt1)[i].y-(*ptrFt0)[i].y,2));
//
//		double d = D/(B*f);
//
//			cv::Vec3f rc;
//			rc[0] = (*ptrFt0)[i].x - u0;
//			rc[1] = (*ptrFt0)[i].y - v0;
//			rc[2] = f;
//
//			cv::Vec3f p = d*rc/cv::norm(rc);
//
//			vctPt.push_back(p);
//	}
//
//	return true;
//}

stereoSolver::~stereoSolver() {
	// TODO Auto-generated destructor stub
}

bool stereoSolver::computeDepth(std::vector<cv::KeyPoint>* ptrFt0,
		std::vector<cv::KeyPoint>* ptrFt1, std::vector<cv::Vec3f>& vctPt) {

	if(u0 == 0 || v0 == 0)
	{
		std::cout<<"Error : Did not specify principle points\n";
		return false;
	}

	if(f == 0 )
	{
		std::cout<<"Error : Did not specify camera focal length\n";
		return false;
	}

	if(B == 0 )
	{
		std::cout<<"Error : Did not specify base line length\n";
		return false;
	}

	if(ptrFt0->size()!=ptrFt1->size())
	{
		std::cout<<"Error : Ft0 size != Ft1 Size\n";
		return false;
	}

	arma::mat dispMat(ptrFt0->size(),1);

	vctPt.clear();

	for(unsigned int i= 0; i < ptrFt0->size(); i++)
	{
		double D = sqrt(pow((*ptrFt1)[i].pt.x-(*ptrFt0)[i].pt.x,2)+pow((*ptrFt1)[i].pt.y-(*ptrFt0)[i].pt.y,2));

		double d = (B*f)/D;

//			cv::Vec3f rc;
//			rc[0] = (*ptrFt0)[i].pt.x - u0;
//			rc[1] = (*ptrFt0)[i].pt.y - v0;
//			rc[2] = f;
//
//			cv::Vec3f p = d*rc/cv::norm(rc);

			float z = d;
			float x = ( (*ptrFt0)[i].pt.x -u0)*z/f;
			float y = ( (*ptrFt0)[i].pt.y -v0)*z/f;

			vctPt.push_back(cv::Vec3f(x,y,z));
	}

	return true;
}
