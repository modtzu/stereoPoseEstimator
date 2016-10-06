/*
 * utility.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: xwong
 */

#include "utility.h"

utility::utility() {
	// TODO Auto-generated constructor stub

}

arma::vec2 utility::objLocToImgLoc(arma::vec3 objLoc, arma::mat camRot,
		arma::vec3 camTrans, arma::mat camIntrin) {

	arma::vec2 imgPt;

	arma::vec3 rLoc = camRot*objLoc + camTrans;

	imgPt[0] = (rLoc[0]/rLoc[2])*camIntrin(0,0) + camIntrin(0,2);
	imgPt[1] = (rLoc[1]/rLoc[2])*camIntrin(1,1) + camIntrin(1,2);

	arma::vec2 imgPtI;

	imgPtI[0] = rint(imgPt[0]);
	imgPtI[1] = rint(imgPt[1]);

	return imgPtI;
}

arma::vec3 utility::cameraRayDir(int px, int py, arma::mat camIntrin) {

	float u = px - camIntrin(0,2);
	float v = py - camIntrin(1,2);

	v = -v;

	cv::Vec3f out(u,v,camIntrin(0,0));

	out = out/norm(out);

	arma::vec3 dir;
	dir<<out[0]<<out[1]<<out[2];

	return dir;
}

arma::vec3 utility::rayRayIntersection(arma::vec3 P0, arma::vec3 D0,
		arma::vec3 P1, arma::vec3 D1) {

	arma::mat A, B, T;

	A 	<< P0[0] - P1[0]<< arma::endr
		<< P0[1] - P1[1]<< arma::endr
		<< P0[2] - P1[2]<< arma::endr;

	B	<< -D0[0] << D1[0] <<arma::endr
		<< -D0[1] << D1[1] <<arma::endr
		<< -D0[2] << D1[2] <<arma::endr;

	T = arma::inv(B.t()*B)*B.t()*A;

	arma::vec3 I0 = P0 + T(0,0)*D0;
	arma::vec3 I1 = P1 + T(1,0)*D1;

	arma::vec3 I = float(0.5)*(I0 + I1);

////	if(debug)
//	{
//		std::cout<<"P"<<P0<<" "<<P1<<"\n";
//		std::cout<<"D"<<D0<<" "<<D1<<"\n";
//		std::cout<<"Ii"<<I0<<" "<<I1<<" "<<I<<"\n\n";
//	}

	return I;

}

cv::Vec3f utility::convolutedIntensity(cv::Point2i pLoc, double sigma,
		cv::Mat img) {
}

arma::vec3 utility::getTanFromNormal(arma::vec3 N, cv::Vec2f dir) {

	/// CV space has -Y axis in relative to GL space?
	cv::Vec3f Dir3d(dir[0],-dir[1],0);

	arma::vec3 aDir;
	aDir<<Dir3d[0]<<Dir3d[1]<<Dir3d[2];

	N/=sqrt(arma::dot(N,N));
	aDir/=sqrt(arma::dot(aDir,aDir));

	arma::vec3 Axis = arma::cross(aDir,N);
	Axis/=sqrt(arma::dot(Axis,Axis));

	arma::vec3 D0 = arma::cross(N,Axis);
	D0/=sqrt(arma::dot(D0,D0));

	return D0;
}

void utility::showHistogram(cv::Mat img) {

	int low = 0;
	int up = 256;

	if(img.channels()==3)
		histRGB(img,low,up);
	else
		histGray(img,low,up);

}

void utility::histRGB(cv::Mat img, int low, int up) {

	cv::vector<cv::Mat> bgrPlanes;
	cv::split(img,bgrPlanes);

	int histPins = 256;

	float range[] = {low,up};
	const float* histRange = {range};

	bool uniform = true;
	bool accumulate = false;

	cv::Mat bHist,rHist,gHist;

	cv::calcHist(&bgrPlanes[0],1,0,cv::Mat(),bHist,1,&histPins,&histRange, uniform, accumulate);
	cv::calcHist(&bgrPlanes[1],1,0,cv::Mat(),gHist,1,&histPins,&histRange, uniform, accumulate);
	cv::calcHist(&bgrPlanes[2],1,0,cv::Mat(),rHist,1,&histPins,&histRange, uniform, accumulate);

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histPins );

	cv::Mat histImg(hist_h,hist_w,CV_8UC3,cv::Scalar(0,0,0));

	cv::normalize(bHist,bHist,0,histImg.rows,cv::NORM_MINMAX,-1,cv::Mat());
	cv::normalize(gHist,gHist,0,histImg.rows,cv::NORM_MINMAX,-1,cv::Mat());
	cv::normalize(rHist,rHist,0,histImg.rows,cv::NORM_MINMAX,-1,cv::Mat());

	for(int i=0; i < histPins; i++)
	{
		cv::line(histImg,cv::Point(bin_w*(i-1), hist_h-cvRound(bHist.at<float>(i-1))),
					     cv::Point(bin_w*(i), hist_h-cvRound(bHist.at<float>(i))), cv::Scalar(255,0,0), 5, 8, 0 );

		cv::line(histImg,cv::Point(bin_w*(i-1), hist_h-cvRound(gHist.at<float>(i-1))),
					     cv::Point(bin_w*(i), hist_h-cvRound(gHist.at<float>(i))), cv::Scalar(0,255,0), 5, 8, 0 );


		cv::line(histImg,cv::Point(bin_w*(i-1), hist_h-cvRound(rHist.at<float>(i-1))),
					     cv::Point(bin_w*(i), hist_h-cvRound(rHist.at<float>(i))), cv::Scalar(0,0,255), 5, 8, 0 );


	}

	cv::imshow("histogram",histImg);

	cv::waitKey(0);

}

void utility::histGray(cv::Mat img, int low, int up) {

	int histPins = 256;

	float range[] = {low,up};
	const float* histRange = {range};

	bool uniform = true;
	bool accumulate = false;

	cv::Mat bHist,rHist,gHist;

	cv::calcHist(&img,1,0,cv::Mat(),bHist,1,&histPins,&histRange, uniform, accumulate);

	std::cout<<bHist<<"\n";

	int hist_w = 720; int hist_h = 720;
	int bin_w = cvRound( (double) hist_w/histPins );

	cv::Mat histImg(hist_h,hist_w,CV_8UC3,cv::Scalar(0,0,0));

	cv::normalize(bHist,bHist,0,histImg.rows,cv::NORM_MINMAX,-1,cv::Mat());

	for(int i=0; i < histPins; i++)
	{
		cv::line(histImg,cv::Point(bin_w*(i-1), hist_h-cvRound(bHist.at<float>(i-1))),
					     cv::Point(bin_w*(i), hist_h-cvRound(bHist.at<float>(i))), cv::Scalar(0,255,255), 5, 8, 0 );

	}

	cv::imshow("histogram",histImg);

	cv::waitKey(0);

}

double utility::FastNoiseVariance(cv::Mat img) {

//	cv::Mat roi = img( cv::Rect(150,50,150,250) );

	cv::Mat kernel = cv::Mat::zeros(3,3,CV_32FC1);

	kernel.at<float>(0,0) = 1; kernel.at<float>(0,1) = -2; kernel.at<float>(0,2) = 1;
	kernel.at<float>(1,0) = -2; kernel.at<float>(1,1) = 4; kernel.at<float>(1,2) = -2;
	kernel.at<float>(2,0) = 1; kernel.at<float>(2,1) = -2; kernel.at<float>(2,2) = 1;

	cv::Mat lapCov;

	int ddepth = -1;
	double delta = 0;
	cv::Point anchor(-1,1);

	cv::filter2D(img,lapCov,ddepth,kernel,anchor,delta,cv::BORDER_DEFAULT);

//	cv::Mat eql;
//	cv::equalizeHist(lapCov,eql);
//	cv::imshow("eql",eql);
//	histGray(eql);

	cv::multiply(lapCov,lapCov,lapCov);

	double sigma = 1/(36.*(img.cols-2)*(img.rows-2))*cv::sum(lapCov)[0];

	return sigma;
}

double utility::FastNoiseVariance(cv::Mat img, cv::Mat featureMap)
{
	cv::Mat kernel = cv::Mat::zeros(3,3,CV_32FC1);

	kernel.at<float>(0,0) = 1; kernel.at<float>(0,1) = -2; kernel.at<float>(0,2) = 1;
	kernel.at<float>(1,0) = -2; kernel.at<float>(1,1) = 4; kernel.at<float>(1,2) = -2;
	kernel.at<float>(2,0) = 1; kernel.at<float>(2,1) = -2; kernel.at<float>(2,2) = 1;

	cv::Mat lapCov;

	int ddepth = -1;
	double delta = 0;
	cv::Point anchor(-1,1);

	cv::filter2D(img,lapCov,ddepth,kernel,anchor,delta,cv::BORDER_DEFAULT);

	cv::multiply(lapCov,featureMap,lapCov);
//	cv::multiply(lapCov,lapCov,lapCov);

//	double sigma = 1/(36.*(img.cols-2)*(img.rows-2))*cv::sum(lapCov)[0];

	double sigma = sqrt(M_PI/2)*(1/(6.*(img.cols-2)*(img.rows-2)))*cv::sum(abs(lapCov))[0];

	return sigma*sigma;
}

cv::Mat utility::FastNoiseVarianceMap(cv::Mat img, cv::Mat featureMap)
{
	cv::Mat kernel = cv::Mat::zeros(3,3,CV_32FC1);

	kernel.at<float>(0,0) = 1; kernel.at<float>(0,1) = -2; kernel.at<float>(0,2) = 1;
	kernel.at<float>(1,0) = -2; kernel.at<float>(1,1) = 4; kernel.at<float>(1,2) = -2;
	kernel.at<float>(2,0) = 1; kernel.at<float>(2,1) = -2; kernel.at<float>(2,2) = 1;

	cv::Mat lapCov;

	int ddepth = -1;
	double delta = 0;
	cv::Point anchor(-1,1);

	cv::filter2D(img,lapCov,ddepth,kernel,anchor,delta,cv::BORDER_DEFAULT);

	cv::multiply(lapCov,featureMap,lapCov);



	lapCov/=36;

	return lapCov;

}



cv::Mat utility::gConvoluteEdgeDetector(cv::Mat img) {

	cv::Mat Gx = cv::Mat::zeros(3,3,CV_32FC1);
	cv::Mat Gy = cv::Mat::zeros(3,3,CV_32FC1);

	Gx.at<float>(0,0) = -1; Gx.at<float>(0,1) = -2; Gx.at<float>(0,2) = -1;
	Gx.at<float>(1,0) = 0; Gx.at<float>(1,1) = 0; Gx.at<float>(1,2) = 0;
	Gx.at<float>(2,0) = 1; Gx.at<float>(2,1) = 2; Gx.at<float>(2,2) = 1;

	Gy.at<float>(0,0) = -1; Gy.at<float>(0,1) = 0; Gy.at<float>(0,2) = 1;
	Gy.at<float>(1,0) = -2; Gy.at<float>(1,1) = 0; Gy.at<float>(1,2) = 2;
	Gy.at<float>(2,0) = -1; Gy.at<float>(2,1) = 0; Gy.at<float>(2,2) = 1;

	cv::Mat cGx, cGy;

	int ddepth = -1;
	double delta = 0;
	cv::Point anchor(-1,1);

	cv::filter2D(img,cGx,ddepth,Gx,anchor,delta,cv::BORDER_DEFAULT);
	cv::filter2D(img,cGy,ddepth,Gy,anchor,delta,cv::BORDER_DEFAULT);

	cv::Mat G = cv::abs(cGx) + cv::abs(cGy);

//	cv::threshold(G,G,240,255,CV_THRESH_TOZERO);

//	cv::imshow("edge",G);
//
//	cv::waitKey(10);
//
//	histGray(G,20,256);

	return G;

}

arma::vec4 utility::RtoQ(arma::mat R) {

	arma::vec4 Q;
	double R11 = R(0,0);	double R12 = R(0,1); 	double R13 = R(0,2);
	double R21 = R(1,0);	double R22 = R(1,1); 	double R23 = R(1,2);
	double R31 = R(2,0);	double R32 = R(2,1); 	double R33 = R(2,2);



	if(R(1,1)>-R(2,2) && R(0,0)> -R(1,1) && R(0,0)>-R(2,2))
	{
		Q(0) = sqrt(1 + R(0,0) + R(1,1) + R(2,2));
		Q(1) = (R23-R32)/sqrt(1+R11 + R22 + R33);
		Q(2) = (R31-R13)/sqrt(1+R11 + R22 + R33);
		Q(3) = (R12-R21)/sqrt(1+R11 + R22 + R33);
	}
	else
	if(R22 < -R33 && R11 > R22 && R11 > R33)
	{
		Q(0) = (R23-R32)/sqrt(1+R11 - R22 - R33);
		Q(1) = sqrt(1+R11 - R22 - R33);
		Q(2) = (R12+R21)/sqrt(1+R11 - R22 - R33);
		Q(3) = (R31+R13)/sqrt(1+R11 - R22 - R33);
	}
	else
	if(R22 > R33 && R11 < R22 && R11 < - R33)
	{
		Q(0) = (R31-R13)/sqrt(1 -R11 + R22 - R33);
		Q(1) = (R12+R21)/sqrt(1 -R11 + R22 - R33);
		Q(2) = sqrt(1 -R11 + R22 - R33);
		Q(2) = (R23+R32)/sqrt(1 -R11 + R22 - R33);
	}
	else
	if(R22 < R33 && R11 < -R22 && R11 < R33)
	{
		Q(0) = (R31-R13)/sqrt(1 -R11 + R22 - R33);
		Q(1) = (R12+R21)/sqrt(1 -R11 + R22 - R33);
		Q(2) = sqrt(1 -R11 + R22 - R33);
		Q(2) = (R23+R32)/sqrt(1 -R11 + R22 - R33);
	}

	Q*=0.5;

	return Q;
}


arma::mat utility::QtoR(arma::vec4 Q) {

	arma::mat R(3,3);

	R(0,0) = Q(0)*Q(0) +Q(1)*Q(1)-Q(2)*Q(2)-Q(3)*Q(3);
	R(0,1) = 2*Q(1)*Q(2)+2*Q(0)*Q(3);
	R(0,2) = 2*Q(1)*Q(3)-2*Q(0)*Q(2);

	R(1,0) = 2*Q(1)*Q(2) - 2*Q(0)*Q(3);
	R(1,1) = Q(0)*Q(0) -Q(1)*Q(1)+Q(2)*Q(2)-Q(3)*Q(3);
	R(1,2) = 2*Q(2)*Q(3) + 2*Q(0)*Q(1);

	R(2,0) = 2*Q(1)*Q(3) + 2*Q(0)*Q(2);
	R(2,1) = 2*Q(2)*Q(3) - 2*Q(0)*Q(1);
	R(2,2) == Q(0)*Q(0) + Q(1)*Q(1) - Q(2)*Q(2) + Q(3)*Q(3);

	return R;
}


utility::~utility() {
	// TODO Auto-generated destructor stub
}

double utility::getLocalDynRng(cv::Mat img) {

	img.convertTo(img,CV_32FC1);

	float avg = sum(img)[0]/(img.cols*img.rows);

	cv::Mat diff = img - avg*cv::Mat::ones(img.size(),CV_32FC1);

	cv::multiply(diff,diff,diff);

	float sigma = sum(diff)[0]/(img.cols*img.rows);

	return (double)3*sqrt(sigma)/255.;

}

cv::RotatedRect utility::plotCovmat(double chisquare_val, cv::Point2f mean, arma::mat sigSing) {

	cv::Mat covMat = cv::Mat_<double>(2,2);
	covMat.at<double>(0,0) = sigSing(0,0);
	covMat.at<double>(1,0) = sigSing(1,0);
	covMat.at<double>(0,1) = sigSing(0,1);
	covMat.at<double>(1,1) = sigSing(1,1);


	//Get the eigenvalues and eigenvectors
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(covMat, true, eigenvalues, eigenvectors);

	//Calculate the angle between the largest eigenvector and the x-axis
	double angle = fabs(atan2(eigenvectors.at<double>(0,1), eigenvectors.at<double>(0,0)));

	//Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
	if(angle < 0)
		angle += 6.28318530718;

	//Conver to degrees instead of radians
	angle = 180*angle/3.14159265359;

	//Calculate the size of the minor and major axes
	double halfmajoraxissize=chisquare_val*sqrt(eigenvalues.at<double>(0));
	double halfminoraxissize=chisquare_val*sqrt(eigenvalues.at<double>(1));

	//Return the oriented ellipse
	//The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
	return cv::RotatedRect(mean, cv::Size2f(halfmajoraxissize, halfminoraxissize), -angle);
}
