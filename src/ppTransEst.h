/*
 * ppTransEst.h
 *
 *  Created on: Sep 13, 2016
 *      Author: xwong
 */

/*
 * Estimate relative transformation between 2 set of point cloud
 */

#include <armadillo>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>

#include <vector>

#ifndef PPTRANSEST_H_
#define PPTRANSEST_H_

class ppTransEst {
private:

	/*
	 *@note estimate parameter vector
	 *@note X = [q1, q2, q3, t1, t2, t3]
	 */

	/// number of points
	int nP;

	/// Measurement : Vec( P(k+1) + P(k) )
	arma::mat Pk;

	/// Measurement : Vec( P(k+1) - P(k) )
	arma::mat Pm;

public:
	ppTransEst();

	/*
	 *@note solve for measurement equation
	 *@input X estimate parameter vector
	 *@output measurement equation output vector
	 */
	arma::mat system(arma::mat X);

	/*
	 *@note estimate relative translation and rotation between 2 set of point cloud
	 *@input Pt0 origin point cloud
	 *@input Pt1 transformed point cloud
	 *@input rotMat output estimate rotation matrix
	 *@input transT output estimate translation vector
	 */
	void solve(std::vector<cv::Vec3f> Pt0, std::vector<cv::Vec3f> Pt1, arma::mat& rotMat, arma::mat& transT);

	/*
	 *@note estimate relative translation and rotation between 2 set of point cloud with ransac eject option
	 *@input Pt0 origin point cloud
	 *@input Pt1 transformed point cloud
	 *@input rotMat output estimate rotation matrix
	 *@input transT output estimate translation vector
	 *@input ransacEjectID vector of eject ID
	 *@return false if require ejection
	 */
	bool solve(std::vector<cv::Vec3f> Pt0, std::vector<cv::Vec3f> Pt1, arma::mat& rotMat, arma::mat& transT, std::vector<int>& ransacEjectID);


	/*
	 *@note estimate relative translation and rotation with linear least square formulation
	 *@input Pt0 origin point cloud
	 *@input Pt1 transformed point cloud
	 *@input rotMat output estimate rotation matrix
	 *@input transT output estimate translation vector
	 *@input ransacEjectID vector of eject ID
	 *@return false if require ejection
	 */
	bool solveLinear(std::vector<cv::Vec3f> Pt0, std::vector<cv::Vec3f> Pt1, arma::mat& rotMat, arma::mat& transT, std::vector<int>& ransacEjectID);

	/*
	 *@note estimate relative translation and rotation from k to k+1 with zero accel model  with linear least square formulation
	 *@input Pt0 k-1 point cloud
	 *@input Pt1 k frame point cloud
	 *@input Pt2 k+1 frame point cloud
	 *@input rotMat output estimate rotation matrix
	 *@input transT output estimate translation vector
	 *@input ransacEjectID vector of eject ID
	 *@return false if require ejection
	 */
	bool solveLinear2ndOrder(std::vector<cv::Vec3f> Pt0, std::vector<cv::Vec3f> Pt1, std::vector<cv::Vec3f> Pt2, arma::mat& rotMat, arma::mat& transT, std::vector<int>& ransacEjectID);

	/*
	 *@note estimate relative translation and rotation from k to k+1 with zero zerg model with linear least square formulation
	 *@input Pt0 k-2  point cloud
	 *@input Pt1 k-1 point cloud
	 *@input Pt2 k point cloud
	 *@input Pt3 k+1 point cloud
	 *@input rotMat output estimate rotation matrix
	 *@input transT output estimate translation vector
	 *@input ransacEjectID vector of eject ID
	 *@return false if require ejection
	 */
	bool solveLinear3rdOrder(std::vector<cv::Vec3f> Pt0, std::vector<cv::Vec3f> Pt1, std::vector<cv::Vec3f> Pt2 , std::vector<cv::Vec3f> Pt3, arma::mat& rotMat, arma::mat& transT, std::vector<int>& ransacEjectID);


	/*
	 *@note conversion from CRP to rotation matrix
	 *@input q input CRP
	 *@output 3X3 rotation matrix
	 */
	arma::mat CRP2ROT(arma::mat q);

	/*
	 *@note compute Jacobian matrix
	 *@input X input vector
	 *@output jacobian matrix
	 */
	arma::mat jacobianMatix(arma::mat X);

	/*
	 *@note cross product matrix
	 */
	arma::mat crossProductMatrix(arma::vec x);

	virtual ~ppTransEst();
};

#endif /* PPTRANSEST_H_ */
