/*
 * kltUncertainty.cpp
 *
 *  Created on: Aug 24, 2015
 *      Author: xwong
 */

#include "kltUncertainty.h"

kltUncertainty::kltUncertainty() {
	// TODO Auto-generated constructor stub

	cv::Size kernelSize(16,16);

	dGXk = dGdXkernel(kernelSize,1);
	dGYk = dGdYkernel(kernelSize,1);
	dGXXk = ddGdXXkernel(kernelSize,1);
	dGXYk = ddGdXYkernel(kernelSize,1);
	dGYYk = ddGdYYkernel(kernelSize,1);

}


cv::Mat kltUncertainty::LoG(cv::Mat img, cv::Size kernelSize,
		double sigma) {

	cv::Mat kernel(kernelSize,CV_64FC1);

	for(int i = 0 ; i < kernelSize.width; i++)
		for(int j = 0 ; j < kernelSize.height; j++)
		{
			cv::Point2f Pt(i - kernelSize.width/2,j - kernelSize.height/2);

			kernel.at<double>(j,i) = LoGkernel(Pt,sigma);
		}

	cv::Mat out;

	cv::filter2D(img,out,-1,kernel);

	return out;

}

double kltUncertainty::LoGkernel(cv::Point2f Pt, double sigma) {

	double D = Pt.dot(Pt);

	double LoG =-(1/(M_PI*pow(sigma,4)))*(1 - D/(2*sigma*sigma))*exp(-D/(2*sigma*sigma));

	return LoG;
}


arma::mat kltUncertainty::kltCovLS(cv::Mat img0, cv::Mat img1,
		cv::Size winSize, cv::KeyPoint Pt0,
		cv::KeyPoint Pt1,arma::mat covPre, cv::Size kernelSize, double sigma) {

		cv::Point2f Ft0 = Pt0.pt;
		cv::Point2f Ft1 = Pt1.pt;

		cv::Point2f C1(Ft0.x - winSize.width/2,Ft0.y - winSize.height/2);
		cv::Point2f C2(Ft1.x - winSize.width/2,Ft1.y - winSize.height/2);

		if(C1.x<0 || C2.x<0 || C1.y <0 ||
				C2.y <0 ||
				C1.x + winSize.width>img0.cols ||
				C2.x + winSize.width>img1.cols||
			    C1.y + winSize.height >img0.rows ||
				C2.y + winSize.height >img1.rows)
		{
			arma::mat I(2,2);
			I.ones();
			I*= -1;
			return I;
		}

		cv::Mat sub0, sub1, sub2;
		sub0 = img0(cv::Rect(C1,winSize));
		sub1 = img1(cv::Rect(C2,winSize));
		sub2 = img1(cv::Rect(C2,winSize));

//		sub0 = img0;//(cv::Rect(C1,winSize));
//		sub1 = img1;//(cv::Rect(C1,winSize));
//		sub2 = img1;//(cv::Rect(C2,winSize));

		sub0.convertTo(sub0,CV_32FC1,1./255,0);
		sub1.convertTo(sub1,CV_32FC1,1./255,0);
		sub2.convertTo(sub2,CV_32FC1,1./255,0);

		cv::Mat dG = sub0 - sub1;

		cv::Mat G1X, G1Y, G0X, G0Y;
		cv::Mat H0X, H0Y, H1X, H1Y, H0XY, H1XY, H1YX;
		cv::Mat ddGx;
		cv::Mat ddGy;

		cv::filter2D(dG,ddGx,-1,dGXk);
		cv::filter2D(dG,ddGy,-1,dGYk);

		cv::filter2D(sub1,G1X,-1,dGXk);
		cv::filter2D(sub1,G1Y,-1,dGYk);

		cv::filter2D(sub0,G0X,-1,dGXk);
		cv::filter2D(sub0,G0Y,-1,dGYk);

		cv::filter2D(sub0,H0X,-1,dGXXk);
		cv::filter2D(sub0,H0Y,-1,dGYYk);
		cv::filter2D(sub0,H0XY,-1,dGXYk);

		cv::filter2D(sub1,H1X,-1,dGXXk);
		cv::filter2D(sub1,H1Y,-1,dGYYk);
		cv::filter2D(sub1,H1XY,-1,dGXYk);
		H1YX = H1XY;

		arma::mat A(winSize.width*winSize.height,2);
		arma::mat B(winSize.width*winSize.height,1);

		arma::mat dBdu(winSize.width*winSize.height,1);
		arma::mat dAduT(winSize.width*winSize.height,1);

		arma::mat dBdv(winSize.width*winSize.height,1);
		arma::mat dAdvT(winSize.width*winSize.height,1);

		cv::Point2f T = Ft1 - Ft0;

		{
		arma::mat A(2,2);
		arma::mat B(2,1);

		A.zeros();
		B.zeros();

		for(int xi =0 ; xi < winSize.width; xi++)
			for(int yi =0 ; yi < winSize.height; yi++)
			{
				A(0,0) += G1X.at<float>(yi,xi)*G1X.at<float>(yi,xi);
				A(0,1) += G1X.at<float>(yi,xi)*G1Y.at<float>(yi,xi);
				A(1,0) += G1Y.at<float>(yi,xi)*G1X.at<float>(yi,xi);
				A(1,1) += G1Y.at<float>(yi,xi)*G1Y.at<float>(yi,xi);

				B(0,0) += (float)dG.at<float>(yi,xi)*G1X.at<float>(yi,xi); /// may be a constant gain?
				B(1,0) += (float)dG.at<float>(yi,xi)*G1Y.at<float>(yi,xi);
			}

			arma::mat D = A.i()*B;
			T.x = D(0,0);
			T.y = D(1,0);

		}


//		cv::Mat Err0 = sub0 - (sub1 + G1X*T.x + G1Y*T.y);
//		cv::Mat Err1 = sub2 - sub0;

		arma::mat sigmaUVM(2,2);
		sigmaUVM.zeros();

		arma::mat Wx(winSize.width*winSize.height,winSize.width*winSize.height);
		arma::mat Wy;
		Wx.zeros();
		Wy = Wx;

		for(int xi =0 ; xi < winSize.width; xi++)
			for(int yi =0 ; yi < winSize.height; yi++)
			{

				A(yi+xi*winSize.height,0) = G1X.at<float>(yi,xi);
				A(yi+xi*winSize.height,1) = G1Y.at<float>(yi,xi);

				B(yi+xi*winSize.height,0) = (float)dG.at<float>(yi,xi);

				dBdu(yi+xi*winSize.height,0) = G0X.at<float>(yi,xi) - G1X.at<float>(yi,xi);
				dBdv(yi+xi*winSize.height,0) = G0Y.at<float>(yi,xi) - G1Y.at<float>(yi,xi);

				dAduT(yi+xi*winSize.height,0) = T.x*H1X.at<float>(yi,xi) + T.y*H1YX.at<float>(yi,xi);
				dAdvT(yi+xi*winSize.height,0) = T.x*H1XY.at<float>(yi,xi) + T.y*H1Y.at<float>(yi,xi);

				double x = xi - winSize.width/2;
				double y = yi - winSize.height/2;

				double sigma = winSize.height/2;

				if(Pt0.size < winSize.height)
					sigma = Pt0.size/2;

				double w = Gaussian(x,y,sigma);

				arma::mat UV(2,1);
				UV(0,0) = x;
				UV(1,0) = y;

				sigmaUVM += w*UV*UV.t();

//				Wx(yi+xi*winSize.height,yi+xi*winSize.height) = Gaussian(x,y,2);
//				Wy(yi+xi*winSize.height,yi+xi*winSize.height) = Gaussian(x,y,2);

				Wx(yi+xi*winSize.height,yi+xi*winSize.height) = (1/max(float(1),fabs(G0X.at<float>(yi,xi) - G1X.at<float>(yi,xi))));
				Wy(yi+xi*winSize.height,yi+xi*winSize.height) = (1/max(float(1),fabs(G0Y.at<float>(yi,xi) - G1Y.at<float>(yi,xi))));

			}

		Wx.eye();
		Wy.eye();

		if(arma::det(A.t()*A)<1e-4)
			return covPre.zeros();

		arma::mat dTdu = arma::inv((A.t()*Wx*A))*A.t()*Wx*(dBdu - dAduT);
		arma::mat dTdv = arma::inv((A.t()*Wy*A))*A.t()*Wy*(dBdv - dAdvT);

		arma::mat K(2,2);

		K.col(0) = arma::abs(dTdu.col(0));
		K.col(1) = arma::abs(dTdv.col(0));

		arma::mat sigmaUV(2,2);
		sigmaUV = sigmaUVM;

//		sigmaUV.eye();

		if(arma::det(covPre) != 0)
			sigmaUV = covPre;

		arma::mat Bx = dBdu - dAduT;
		arma::mat By = dBdv - dAdvT;

		int mode = 0;

		arma::mat sigmaKLT;

		switch(mode)
		{
			case 0:
			{
				arma::mat X2 = K*sigmaUV + sigmaUV*K.t();
				arma::mat X =  K*(sigmaUV)*K.t()+ X2;

				arma::cx_vec eigVal;
				arma::cx_mat eigVec;

				arma::eig_gen(eigVal,eigVec,X);

				arma::mat eig = arma::real(eigVal);

//				eig.print("eig");

				if(eig(0,0)>1e2 ||eig(1,0)>1e2)
				{
					sigmaKLT.eye(2,2);
					sigmaKLT*= -1;
				}
				else
					sigmaKLT = (X + sigmaUV);

				break;
			}
			case 1: /// maximum covariance
			{
				arma::mat X =  K*(sigmaUV)*K.t();
				sigmaKLT = 2*(X + sigmaUV);

				if(sigmaKLT(0,0) <0 || sigmaKLT(1,1) <0 || sigmaKLT(0,0)>1e4 ||sigmaKLT(1,1)>1e4)
				{
					sigmaKLT.zeros(2,2);
				}

				break;
			}
			case 2:
			{
			}

		}

		return sigmaKLT;

}

arma::mat kltUncertainty::getSigmaUV(cv::Mat img0, cv::Mat img1, cv::Point2f Ft0,
		cv::Point2f Ft1,cv::Size KLTwinSize, cv::Mat G1X, cv::Mat G1Y ) {
	arma::mat sigmaUV(2,2);
	sigmaUV.eye();

	cv::Mat dG = img1 - img0;

	cv::Mat Hu(KLTwinSize,CV_32FC1);
	cv::Mat Hv(KLTwinSize,CV_32FC1);

	for(int xi = 0; xi<KLTwinSize.width; xi++)
		for(int yi = 0; yi<KLTwinSize.height; yi++)
		{
			float dg = (dG.at<uchar>(yi,xi));

			float hu = dg/(G1X.at<float>(yi, xi));
			float hv = dg/(G1Y.at<float>(yi, xi));

			if(hu != hu || isinf(hu)) hu = 10;//KLTwinSize.width;
			if(hv != hv || isinf(hv)) hv = 10;//KLTwinSize.height;

			sigmaUV(0,0) += hu*hu;
			sigmaUV(1,0) += hu*hv;
			sigmaUV(0,1) += hu*hv;
			sigmaUV(1,1) += hv*hv;
		}

	sigmaUV/= (KLTwinSize.height*KLTwinSize.width);

	return sigmaUV;
}

arma::mat kltUncertainty::getSigmaU(cv::Size KLTwinSize, cv::Mat G0X, cv::Mat G0Y) {

	arma::mat sigmaUV(2,2);
//	sigmaUV.eye();

	sigmaUV.zeros();

	for(int xi = 0; xi<KLTwinSize.width; xi++)
		for(int yi = 0; yi<KLTwinSize.height; yi++)
		{
			float hu = (G0X.at<float>(yi, xi));
			float hv = (G0Y.at<float>(yi, xi));

			double x = xi - KLTwinSize.width/2;
			double y = yi - KLTwinSize.height/2;
			double sigma = sqrt(KLTwinSize.width*KLTwinSize.width + KLTwinSize.height*KLTwinSize.height);

			double wi = Gaussian(x,y,sigma);

			sigmaUV(0,0) += wi*hu*hu;
			sigmaUV(1,0) += wi*hu*hv;
			sigmaUV(0,1) += wi*hv*hu;
			sigmaUV(1,1) += wi*hv*hv;
		}

//	sigmaUV/= KLTwinSize.area();

	sigmaUV.print("sigUV0");

	sigmaUV = sigmaUV.i();

	return sigmaUV;

}

cv::Mat kltUncertainty::dGdXkernel(cv::Size kernelSize, double sigma) {

	cv::Mat kernel(kernelSize,CV_32FC1);

	float sigma2 = sigma*sigma;
	float sigma4 = sigma2*sigma2;

	for(int xi =0; xi<kernelSize.width; xi++)
		for(int yi =0; yi<kernelSize.height; yi++)
		{
			float x = xi - kernelSize.width/2;
			float y = yi - kernelSize.height/2;

			float L = (x*x+y*y);

			kernel.at<float>(yi,xi) = (-x/(2*M_PI*sigma4))*exp(-L/(2*sigma2));
		}

	return kernel;

}

cv::Mat kltUncertainty::dGdYkernel(cv::Size kernelSize, double sigma) {

	cv::Mat kernel(kernelSize,CV_32FC1);

	float sigma2 = sigma*sigma;
	float sigma4 = sigma2*sigma2;

	for(int xi =0; xi<kernelSize.width; xi++)
		for(int yi =0; yi<kernelSize.height; yi++)
		{
			float x = xi - kernelSize.width/2;
			float y = yi - kernelSize.height/2;

			float L = (x*x+y*y);

			kernel.at<float>(yi,xi) = (-y/(2*M_PI*sigma4))*exp(-L/(2*sigma2));
		}

	return kernel;
}

cv::Mat kltUncertainty::ddGdXXkernel(cv::Size kernelSize, double sigma) {

	cv::Mat kernel(kernelSize,CV_32FC1);

	float sigma2 = sigma*sigma;
	float sigma6 = sigma2*sigma2*sigma2;

	for(int xi =0; xi<kernelSize.width; xi++)
		for(int yi =0; yi<kernelSize.height; yi++)
		{
			float x = xi - kernelSize.width/2;
			float y = yi - kernelSize.height/2;

			float L = (x*x+y*y);

			kernel.at<float>(yi,xi) = ((x*x - sigma2)/(2*M_PI*sigma6))*exp(-L/(2*sigma2));
		}

	return kernel;
}

cv::Mat kltUncertainty::ddGdYYkernel(cv::Size kernelSize, double sigma) {
	cv::Mat kernel(kernelSize,CV_32FC1);

	float sigma2 = sigma*sigma;
	float sigma6 = sigma2*sigma2*sigma2;

	for(int xi =0; xi<kernelSize.width; xi++)
		for(int yi =0; yi<kernelSize.height; yi++)
		{
			float x = xi - kernelSize.width/2;
			float y = yi - kernelSize.height/2;

			float L = (x*x+y*y);

			kernel.at<float>(yi,xi) = ((y*y - sigma2)/(2*M_PI*sigma6))*exp(-L/(2*sigma2));
		}

	return kernel;
}

cv::Mat kltUncertainty::ddGdXYkernel(cv::Size kernelSize, double sigma) {

	cv::Mat kernel(kernelSize,CV_32FC1);

	float sigma2 = sigma*sigma;
	float sigma6 = sigma2*sigma2*sigma2;

	for(int xi =0; xi<kernelSize.width; xi++)
		for(int yi =0; yi<kernelSize.height; yi++)
		{
			float x = xi - kernelSize.width/2;
			float y = yi - kernelSize.height/2;

			float L = (x*x+y*y);

			kernel.at<float>(yi,xi) = ((x*y)/(2*M_PI*sigma6))*exp(-L/(2*sigma2));
		}

	return kernel;
}

double kltUncertainty::Gaussian(double x, double y, double sigma) {

	double sigma2 = sigma*sigma;

	double w = 1/(2*M_PI*sigma2)*exp(-(x*x + y*y)/(2*sigma2));

	return w;
}

kltUncertainty::~kltUncertainty() {
	// TODO Auto-generated destructor stub
}

arma::mat kltUncertainty::dTdxyClose(cv::Mat G1X, cv::Mat G1Y,cv::Mat G0X, cv::Mat G0Y,
		cv::Mat H1X, cv::Mat H1Y, cv::Mat H1XY, cv::Mat sub0, cv::Mat sub1) {

	int numEle = G1X.cols*G1X.rows;

	cv::Mat Iu = G1X.reshape(1,numEle);
	cv::Mat Iv = G1Y.reshape(1,numEle);
	cv::Mat Iuu = H1X.reshape(1,numEle);
	cv::Mat Iuv = H1XY.reshape(1,numEle);
	cv::Mat Ivv = H1Y.reshape(1,numEle);

	cv::Mat Ju = G0X.reshape(1,numEle);
	cv::Mat Jv = G0Y.reshape(1,numEle);

	cv::Mat J = sub0.reshape(1,numEle);
	cv::Mat I = sub1.reshape(1,numEle);

	arma::mat A(2,2);
	A(0,0) = Iu.dot(Iu);
	A(0,1) = Iu.dot(Iv);
	A(1,0) = Iv.dot(Iu);
	A(1,1) = Iv.dot(Iv);

	arma::mat B(2,1);
	B(0,0) = Iu.dot(J -I);
	B(1,0) = Iv.dot(J -I);

	arma::mat AtA = A.t()*A;
	double D = arma::det(AtA);

	arma::mat AtAi = AtA.i()*D;

	float dDdU = 2*Iuu.dot(Iu)*Iv.dot(Iv) + Iu.dot(Iu)*(2*Iuv.dot(Iv)) - 2*Iu.dot(Iv)*(Iuu.dot(Iv) + Iu.dot(Iuv));
	dDdU/=(-D*D);

	float dDdV = 2*Iuv.dot(Iu)*Iv.dot(Iv) + Iu.dot(Iu)*(2*Ivv.dot(Iv)) - 2*Iu.dot(Iv)*(Iuv.dot(Iv) + Iu.dot(Ivv));
	dDdV/=(-D*D);

	arma::mat dAtAiMDu(2,2);
	dAtAiMDu(0,0) = 2*(Iv.dot(Iuv));
	dAtAiMDu(0,1) = -(Iuu.dot(Iv) + Iu.dot(Iuv));
	dAtAiMDu(1,0) = -(Iuv.dot(Iu) + Iv.dot(Iuu));
	dAtAiMDu(1,1) = 2*Iuu.dot(Iu);

	arma::mat DAtAiDu = dDdU*AtAi + (1/D)*(dAtAiMDu);

	arma::mat dAtAiMDv(2,2);
	dAtAiMDv(0,0) = 2*Iv.dot(Ivv);
	dAtAiMDv(0,1) = -(Iuv.dot(Iv) + Iu.dot(Ivv));
	dAtAiMDv(1,0) = -(Ivv.dot(Iu) + Iv.dot(Iuv));
	dAtAiMDv(1,1) = 2*Iuv.dot(Iu);

	arma::mat DAtAiDv = dDdV*AtAi + (1/D)*(dAtAiMDv);

	arma::mat DAtbDu(2,1);
	DAtbDu(0,0) = Iuu.dot(J - I) + Iu.dot(Ju - Iu);
	DAtbDu(0,0) = Iuv.dot(J - I) + Iv.dot(Ju - Iu);

	arma::mat DAtbDv(2,1);
	DAtbDv(0,0) = Iuv.dot(J - I) + Iu.dot(Jv - Iv);
	DAtbDv(0,0) = Ivv.dot(J - I) + Iv.dot(Jv - Iv);

	arma::mat dTdu = DAtAiDu*B + (AtA).i()*DAtbDu;

	arma::mat dTdv = DAtAiDv*B + (AtA).i()*DAtbDv;

	arma::mat dTduv(2,2);
	dTduv.col(0) = dTdu.col(0);
	dTduv.col(1) = dTdv.col(0);

	return dTduv;


}

arma::mat kltUncertainty::kAsTotalLS(arma::mat A, arma::mat Bx, arma::mat By, arma::mat D) {

	arma::mat AtAi = (A.t()*A).i();

	arma::mat Xx(A.n_rows,A.n_cols + Bx.n_cols);
	arma::mat Xy(A.n_rows,A.n_cols + Bx.n_cols);

	Xx.submat(0,0,A.n_rows-1,A.n_cols-1) = A;
	Xx.submat(0,A.n_cols,A.n_rows-1,Xx.n_cols-1) = Bx;

	Xy.submat(0,0,A.n_rows-1,A.n_cols-1) = A;
	Xy.submat(0,A.n_cols,A.n_rows-1,Xx.n_cols-1) = By;

	arma::mat U,V;
	arma::vec Sx, Sy;

	arma::svd(U,Sx,V,Xx);
	arma::svd(U,Sy,V,Xx);

	arma::mat I(2,2);
	I.eye();

	arma::mat ckx = arma::inv(A.t()*A + I*Sx(2,0)*Sx(2,0))*A.t()*Bx;
	arma::mat cky = arma::inv(A.t()*A + I*Sy(2,0)*Sy(2,0))*A.t()*By;

	arma::mat K(2,2);
	K.col(0) = ckx.col(0);
	K.col(1) = cky.col(0);

	return K;
}
