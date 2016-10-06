/*
 * imgFtRelated.cpp
 *
 *  Created on: Jun 23, 2015
 *      Author: xwong
 */

#include "imgFtRelated.h"

///
///
void imgFtRelated::getFeature(cv::Mat img, vector<cv::KeyPoint>& Ft,
		cv::Mat& descriptor, int flg) {

//	flg = 0;
	cv::Mat gray;
	if(img.channels()>1)
		cvtColor(img,gray,CV_BGR2GRAY);
	else
		img.copyTo(gray);

	if(flg == 0)
	{
		std::vector<cv::KeyPoint> vctKeyPoint;

		ptrFeatureDetector->detect(gray, vctKeyPoint);

		std::vector<cv::KeyPoint> vctRmSKeyPoint;

		for(size_t i =0 ; i < vctKeyPoint.size(); i++)
		{
			bool sameFlg = false;

			for(size_t j =0; j < vctRmSKeyPoint.size(); j++)
			{
				cv::Point2f Err = vctRmSKeyPoint[j].pt - vctKeyPoint[i].pt;

				if(Err.dot(Err)<1)
				{
					sameFlg = true;
					break;
				}
			}

			if(!sameFlg &&
					gray.at<uchar>(vctKeyPoint[i].pt.y,vctKeyPoint[i].pt.x)>0)// &&
	//					harrisCornerCheck(vctKeyPoint[i].pt,img))
				vctRmSKeyPoint.push_back(vctKeyPoint[i]);

		}

		ptrFeatureDetector->compute(img, vctRmSKeyPoint, descriptor);

		Ft = vctRmSKeyPoint;
	}
	else
	{
		vector<cv::KeyPoint> FtG;
		getFeatureGdtt(gray, FtG);

		Ft = FtG;
	}
}

///
///

void imgFtRelated::getDescriptorForPt(cv::Mat img,
		vector<cv::KeyPoint> kFt, cv::Mat& descriptor) {

	ptrFeatureDetector->compute(img, kFt,descriptor);
}

///
///


bool imgFtRelated::matchFeature(cv::Mat img1, cv::Mat img2,
		vector<cv::KeyPoint>& kFt1, vector<cv::KeyPoint>& kFt2,
		vector<cv::Point2f>& Ft0, vector<cv::Point2f>& Ft1, cv::Mat descriptor1,
		cv::Mat descriptor2) {

		cv::FlannBasedMatcher matcher;

		std::vector<cv::DMatch> matched;

		matcher.match(descriptor1, descriptor2, matched);

		double maxDist = 0;
		double minDist = 100;

		double avgDist = 0;

		for (int i = 0; i < descriptor1.rows; i++) {
			double dist = matched[i].distance;

			avgDist += dist;

			if (dist < minDist)
				minDist = dist;
			if (dist > maxDist)
				maxDist = dist;
		}

		avgDist/=descriptor1.rows;

		double sigma = 0;

		for (int i = 0; i < descriptor1.rows; i++) {
			double dist = matched[i].distance;

			sigma += (dist-avgDist)*(dist-avgDist);
		}

		sigma /= descriptor1.rows+1;

		sigma = sqrt(sigma);

		std::vector<cv::DMatch> goodMatched;

		Ft0.clear();
		Ft1.clear();

		std::vector<cv::KeyPoint> tKpt1, tKpt2;

		double Th = max(2*minDist, 0.02);//avgDist - 1.0*sigma;

		std::deque <cv::DMatch> vctRemDist;

		for (int i = 0; i < descriptor1.rows; i++) {

			if (matched[i].distance <= Th)
			{
				goodMatched.push_back(matched[i]);

				Ft0.push_back(kFt1[matched[i].queryIdx].pt);
				Ft1.push_back(kFt2[matched[i].trainIdx].pt);

				tKpt1.push_back(kFt1[matched[i].queryIdx]);
				tKpt2.push_back(kFt2[matched[i].queryIdx]);

			}
			else
			{
				vctRemDist.push_back(matched[i]);
			}
		}

//		while(Ft0.size()<50)
//		{
//			auto smallest = std::min_element(std::begin(vctRemDist),std::end(vctRemDist));
//			int ID = std::distance(std::begin(vctRemDist),smallest);
//
//			Ft0.push_back(kFt1[vctRemDist[ID].queryIdx].pt);
//			Ft1.push_back(kFt2[vctRemDist[ID].trainIdx].pt);
//
//			tKpt1.push_back(kFt1[matched[ID].queryIdx]);
//			tKpt2.push_back(kFt2[matched[ID].queryIdx]);
//
//			goodMatched.push_back(vctRemDist[ID]);
//
//			vctRemDist.erase(smallest);
//
//			if(vctRemDist.empty())
//				break;
//		}

		if(Ft0.size()==0)
		{
			return false;
		}

		cv::Mat imgMatched;

		cv::Mat img1C, img2C;

		cv::cvtColor(img1,img1C,CV_GRAY2BGR);
		cv::cvtColor(img2,img2C,CV_GRAY2BGR);

//		std::cout<<goodMatched.size()<<"\n";
//
//		cv::drawMatches(img1C, kFt1, img2C, kFt2, goodMatched, imgMatched,
//				cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
//				cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

//		cv::imshow("Good Matched", imgMatched);
//		cv::imwrite("goodMatch.bmp",imgMatched);

//		cv::waitKey(0);

		kFt1.clear();
		kFt2.clear();

		kFt1 = tKpt1;
		kFt2 = tKpt2;

		return true;
}

///
///


void imgFtRelated::trackFeature(cv::Mat img0, cv::Mat img1,
		vector<cv::KeyPoint>& Ft0, vector<cv::KeyPoint>& Ft1, vector<int>& removedID, cv::Size KltWinSize) {

	cv::TermCriteria termCrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);

	std::vector<uchar> status;
	std::vector<float> error;

	cv::Size winSize = KltWinSize;

	cv::Mat gray0, gray1;

	if(img0.channels()>1)
	cvtColor(img0, img0, CV_BGR2GRAY);

	if(img1.channels()>1)
	cvtColor(img1, img1, CV_BGR2GRAY);

	vector<cv::Point2f> Ft0Pt;
	vector<cv::Point2f> Ft1Pt;

	cv::KeyPoint::convert(Ft0,Ft0Pt);

	calcOpticalFlowPyrLK(img0, img1, Ft0Pt, Ft1Pt, status, error, winSize, 5,
			termCrit, 0);

//	cornerSubPix(img1,Ft1Pt,winSize,cv::Size(-1,-1),termCrit);

	cv::Mat D0, D1;
	img0.copyTo(D0);
	img1.copyTo(D1);

	removedID.clear();

	vector<cv::KeyPoint> vctNft0;

	Ft1.clear();

	for (size_t i = 0; i < status.size(); i++) {
		if (status[i])
		{

			cv::KeyPoint kFt1 = Ft0[i];
			kFt1.pt = Ft1Pt[i];

			Ft1.push_back(kFt1);

			vctNft0.push_back(Ft0[i]);
		}
		else
		{
			removedID.push_back(i);
		}
	}

	Ft0.clear();
	Ft0 = vctNft0;
}

void imgFtRelated::saveKeyPoint(vector<cv::KeyPoint> Ft, int ID) {
	if(!ftWriter.is_open())
	{
		ftWriter.open("featurePoint.ft");
	}

	ftWriter << ID<<"\t";

	for(int i = 0; i < Ft.size(); i++)
	{
		ftWriter<<Ft[i].pt.x<<","<<Ft[i].pt.y<<",";

//		arma::mat armaFt;
//		armaFt<<Ft[i].pt.x<<Ft[i].pt.y<<1<<arma::endr;

//		arma::mat normFt = camIntrixMat.i()*armaFt.t();

//		ftWriter<<normFt(0,0)<<","<<normFt(1,0)<<",";
	}

	ftWriter<<"\n";
}

cv::Point2f imgFtRelated::hmAffineKLT(cv::Mat img0, cv::Mat img1,
		cv::Point2f Ft0, cv::Size winSize) {


	cv::Point2f Ft1 = Ft0;

	int cout = 0;

	cv::Point2f FtX = Ft1;

	double resErr = 1e10;

	cv::Point2f C1(Ft0.x - winSize.width/2 -1,Ft0.y - winSize.width/2-1);
	cv::Mat sub0;
	sub0 = img0(cv::Rect(C1,winSize+cv::Size(2,2)));

	double Axx = 0;
	double Axy = 0;
	double Ayx = 0;
	double Ayy = 0;

	std::deque<float> dTx;
	std::deque<float> dTy;

	while(cout < 100)
	{
		cv::Vec2f fp1(Ft1.x-round(Ft1.x),Ft1.y-round(Ft1.y));
		cv::Vec2f fm1(0.);

		if(fp1[0] <0)
		{
			fm1[0] = -fp1[0];
			fp1[0] = 0;
		}

		if(fp1[1] <0)
		{
			fm1[1] = -fp1[1];
			fp1[1] = 0;
		}

		cv::Vec2f fk = cv::Vec2f(1.,1.) - fp1 - fm1;


		cv::Point2f C2(Ft1.x - winSize.width/2 -1 ,Ft1.y - winSize.width/2 -1);

		cv::Mat sub1;

		sub1 = img1(cv::Rect(C2,winSize+cv::Size(2,2)));


		cv::Mat dG = sub0 - sub1;

		cv::Mat S0, S1, dS;

		cv::Mat G1X, G1Y;

		cv::Sobel(sub1,G1X,CV_32FC1,1,0,3);
		cv::Sobel(sub1,G1Y,CV_32FC1,0,1,3);

		cv::Mat G0X, G0Y;

		cv::Sobel(sub0,G0X,CV_32FC1,1,0,3);
		cv::Sobel(sub0,G0Y,CV_32FC1,0,1,3);

		arma::mat D(winSize.width*winSize.height,6);

		arma::mat B(winSize.width*winSize.height,1);

		arma::mat W(winSize.width*winSize.height,winSize.width*winSize.height);
		W.zeros();

		for(int xi = 0; xi < winSize.width; xi++)
			for(int yi = 0 ; yi < winSize.height; yi++)
			{
				int kxi = xi+1;
				int kyi = yi+1;

				float x = xi - winSize.width/2.;
				float y = yi - winSize.height/2.;

				float axf = (1.+Axx)*x + Axy*y;
				float ayf = (Ayx)*x + (1.+Ayy)*y;

				float aXf = (1+Axx)*x + Axy*y + winSize.width/2 +1;
				float aYf = (Ayx)*x + (1+Ayy)*y + winSize.height/2 +1;

				int aX = round(aXf);
				int aY = round(aYf);

				cv::Vec2f fAp1(aXf - aX, aYf - aY);
				cv::Vec2f fAm1(0.);

					if(fAp1[0] <0)
					{
						fAm1[0] = -fAp1[0];
						fAp1[0] = 0;
					}

					if(fAp1[1] <0)
					{
						fAm1[1] = -fAp1[1];
						fAp1[1] = 0;
					}

				cv::Vec2f fAk = cv::Vec2f(1.,1.) - fAm1 - fAp1;

				double dGdX = fAk[0]*G1X.at<float>(aY,aX) + fAp1[0]*G1X.at<float>(aY,aX+1) + fAm1[0]*G1X.at<float>(aY,aX-1);

				double dGdY = fAk[1]*G1Y.at<float>(aY,aX) + fAp1[1]*G1Y.at<float>(aY+1,aX) + fAm1[1]*G1Y.at<float>(aY-1,aX);

				D(yi+xi*winSize.height,0) = dGdX;
				D(yi+xi*winSize.height,1) = dGdY;

				D(yi+xi*winSize.height,2) = axf*dGdX;
				D(yi+xi*winSize.height,3) = ayf*dGdX;

				D(yi+xi*winSize.height,4) = axf*dGdY;
				D(yi+xi*winSize.height,5) = ayf*dGdY;

				float S0x = fAk[0]*(float)sub0.at<uchar>(aY,aX) + fAp1[0]*(float)sub1.at<uchar>(aY,aX+1) + fAm1[0]*(float)sub1.at<uchar>(aY,aX-1);
				float S0y =	fAk[1]*(float)sub0.at<uchar>(aY,aX) + fAp1[1]*(float)sub1.at<uchar>(aY+1,aX) + fAm1[1]*(float)sub1.at<uchar>(aY-1,aX);

				float S0 = 0.5*(S0x+S0y);

				float S1x = fk[0]*(float)sub1.at<uchar>(kyi,kxi) + fp1[0]*(float)sub1.at<uchar>(kyi,kxi+1) + fm1[0]*(float)sub1.at<uchar>(kyi,kxi-1);
				float S1y =	fk[1]*(float)sub1.at<uchar>(kyi,kxi) + fp1[1]*(float)sub1.at<uchar>(kyi+1,kxi) + fm1[1]*(float)sub1.at<uchar>(kyi-1,kxi);

				float S1 = 0.5*(S1x+S1y);

				B(yi+xi*winSize.height,0) = (S0 - S1);

			}

		if(arma::det(D.t()*D)==0)
			break;

		arma::mat T = (D.t()*D).i()*D.t()*B;

		Axx = T(2,0);
		Axy = T(3,0);
		Ayx = T(4,0);
		Ayy = T(5,0);

		cv::Point2f out(T(0,0),T(1,0));

		out += Ft1;

		Ft1 = out;

		if(dTx.size()<5)
		{
			dTx.push_back(T(0,0));
			dTy.push_back(T(1,0));
		}
		else
		{
			float sumX = 0, sumY = 0;

			for(int i=0 ; i <5; i++)
			{
				sumX += dTx[i];
				sumY += dTy[i];
			}

			dTx.pop_front();
			dTy.pop_front();

			dTx.push_back(T(0,0));
			dTy.push_back(T(1,0));

			if(fabs(sumX)<0.1 && fabs(sumY)<0.1)
				break;

		}

		cout ++;
	}


	return Ft1;


}

cv::Point2f imgFtRelated::hmCvKLT(cv::Mat img0, cv::Mat img1, cv::Point2f Ft0,
		cv::Size winSize) {

	cv::Point2f Ft1 = Ft0;

	int cout = 0;

	cv::Point2f FtX = Ft1;

	double resErr = 1e10;

	cv::Point2f C1(Ft0.x - winSize.width/2,Ft0.y - winSize.width/2);
	cv::Mat sub0;

	cv::Point2f Tr(0,0);
	//// check for boundary condition

	testWinSize(img0.size(),C1,winSize);

	sub0 = img0(cv::Rect(C1,winSize));
	sub0.convertTo(sub0,CV_32FC1,1./255,0);


	cv::Mat dGXk = dGdXkernel(cv::Size(5,5),1);
	cv::Mat dGYk = dGdYkernel(cv::Size(5,5),1);

	cv::Mat G1X, G1Y,G1X2, G1Y2;
	cv::Mat G0X, G0Y, G0X2, G0Y2;

	cv::filter2D(sub0,G0X,-1,-dGXk);
	cv::filter2D(sub0,G0Y,-1,-dGYk);

	arma::mat dTduT(2,2);
	dTduT.zeros();

	while(cout < 10)
	{
		cv::Point2f C2(Ft1.x - winSize.width/2,Ft1.y - winSize.width/2);

		cv::Mat sub1;

		sub1 = img1(cv::Rect(C2,winSize));
		sub1.convertTo(sub1,CV_32FC1,1./255,0);

		cv::Mat dG = sub0 - sub1;

		cv::Mat S0, S1, dS;

		cv::filter2D(sub1,G1X,-1,-dGXk);
		cv::filter2D(sub1,G1Y,-1,-dGYk);

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

		A(1,0) = A(0,1);

		if(arma::det(A.t()*A)==0)
			break;

		arma::mat T = (A).i()*B;

		cv::Point2f out(T(0,0),T(1,0));

		out += Ft1;

		Ft1 = out;

		if(fabs(T(0,0)) < 1e-2 && fabs(T(1,0)) < 1e-2)
			break;

		cout ++;
	}

	return Ft1;

}

cv::Mat imgFtRelated::dGdXkernel(cv::Size kernelSize, double sigma) {

		cv::Mat kernel(kernelSize,CV_32FC1);

		float sigma2 = sigma*sigma;
		float sigma4 = sigma2*sigma2;

		for(int xi =0; xi<kernelSize.width; xi++)
			for(int yi =0; yi<kernelSize.height; yi++)
			{
				float x = xi - kernelSize.width/2;
				float y = yi - kernelSize.height/2;

				float L = sqrt(x*x+y*y);

				kernel.at<float>(yi,xi) = -x/(2*M_PI*sigma4)*exp(-L/(2*sigma2));

			}

		return kernel;

}

cv::Mat imgFtRelated::dGdYkernel(cv::Size kernelSize, double sigma) {

		cv::Mat kernel(kernelSize,CV_32FC1);

		float sigma2 = sigma*sigma;
		float sigma4 = sigma2*sigma2;

		for(int xi =0; xi<kernelSize.width; xi++)
			for(int yi =0; yi<kernelSize.height; yi++)
			{
				float x = xi - kernelSize.width/2;
				float y = yi - kernelSize.height/2;

				float L = sqrt(x*x+y*y);

				kernel.at<float>(yi,xi) = -y/(2*M_PI*sigma4)*exp(-L/(2*sigma2));
			}

		return kernel;

}

bool imgFtRelated::testWinSize(cv::Size imgSize, cv::Point2f& C1,
		cv::Size& winSize) {

	bool flg = false;

	if(C1.x<0)
	{
		winSize.width += C1.x;
		C1.x = 0;
		flg = true;
	}

	if(C1.y<0)
	{
		winSize.height += C1.y;
		C1.y = 0;
		flg = true;
	}

	if(C1.x + winSize.width > imgSize.width)
		{
			winSize.width = imgSize.width - C1.x;
			flg = true;
		}

	if(C1.y + winSize.height > imgSize.height)
		{
			winSize.height = imgSize.height - C1.y;
			flg = true;
		}

return flg;
}

bool imgFtRelated::stereoMatch(cv::Mat img0, cv::Mat img1,
		cv::KeyPoint Ft0, cv::KeyPoint& Ft1, cv::Size winSize) {

	Ft1 = Ft0;

	cv::Point2f pt0 = Ft0.pt;

	cv::Point2f C1(pt0.x - winSize.width/2,pt0.y - winSize.height/2);

	cv::Mat img1f, img0f;
	img0.convertTo(img0f,CV_32FC1,1./255,0);
	img1.convertTo(img1f,CV_32FC1,1./255,0);

	cv::Mat warpMat = cv::Mat::zeros(2,3,CV_32FC1);
	warpMat.at<float>(0,0) = 1;
	warpMat.at<float>(1,1) = 1;

	warpMat.at<float>(0,2) = C1.x;
	warpMat.at<float>(1,2) = C1.y;

	cv::Mat refWin;
	cv::warpAffine(img0f,refWin,warpMat,winSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP);

	cv::Mat dGXk = dGdXkernel(cv::Size(7,7),1);
	cv::Mat dGYk = dGdYkernel(cv::Size(7,7),1);

	cv::Mat G1X, G1Y,G1X2, G1Y2, G1Xf, G1Yf;
	cv::Mat G0X, G0Y, G0X2, G0Y2;

	cv::Point2f C2 = C1;

	int k =0;
	while(k < 20)
	{
		warpMat.at<float>(0,2) = C2.x;
		warpMat.at<float>(1,2) = C2.y;

		cv::Mat trgWin;
		cv::warpAffine(img1f,trgWin,warpMat,winSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP);

		cv::Mat G0X;
		cv::filter2D(trgWin,G0X,-1,-dGXk);

		cv::Mat Err = trgWin - refWin;

		double err = cv::norm(Err);
		if(err < 1e-6)
			break;

		double Iu = 0;
		double IuD = 0;

		for(int i =0; i < trgWin.cols; i++)
		{
			Iu += G0X.col(i).dot(G0X.col(i));
			IuD += G0X.col(i).dot(Err.col(i));
		}

		float du = IuD/Iu;

//		std::cout<<err<<" "<<du<<"\n";

		C2.x -= du;
		k++;
	}

	Ft1.pt.x = C2.x+ winSize.width/2;
	Ft1.pt.y = C2.y+ winSize.height/2;

//	std::cout<<"\n";

	return true;

}

imgFtRelated::~imgFtRelated() {
	// TODO Auto-generated destructor stub
}

cv::Point2f imgFtRelated::hmKLT(cv::Mat img0, cv::Mat img1, cv::Point2f Ft0,
		cv::Size winSize) {

	cv::Point2f Ft1 = Ft0;

	int cout = 0;

	cv::Point2f FtX = Ft1;

	double resErr = 1e10;

	cv::Point2f C1(Ft0.x - winSize.width/2,Ft0.y - winSize.width/2);

	cv::Point2f T(0,0);
	//// check for boundary condition
	if(C1.x<0) T.x = C1.x;
	if(C1.y<0) T.y = C1.y;
	if(C1.x + winSize.width >= winSize.width) T.x = winSize.width - C1.x;
	if(C1.y + winSize.height >= winSize.height) T.y = winSize.height - C1.y;

	C1 += T;

	cv::Mat sub0;
	sub0 = img0(cv::Rect(C1,winSize));

	sub0.convertTo(sub0,CV_32FC1,1/255.,0);

	int Tsign = 0;

	while(cout < 1)
	{
		cv::Point2f C2(Ft1.x - winSize.width/2,Ft1.y - winSize.width/2);

		cv::Point2f T2(0,0);
		//// check for boundary condition
		if(C2.x<0) T.x = C2.x;
		if(C2.y<0) T.y = C2.y;
		if(C2.x + winSize.width >= winSize.width) T.x = winSize.width - C2.x;
		if(C2.y + winSize.height >= winSize.height) T.y = winSize.height - C2.y;

		C2 += T;

		cv::Mat sub1;

		sub1 = img1(cv::Rect(C2,winSize));

		sub1.convertTo(sub1,CV_32FC1,1/255.,0);

		cv::Mat dG = sub0 - sub1;

		cv::Mat S0, S1, dS;

		cv::Mat G1X, G1Y,G1X2, G1Y2;
		cv::Mat G0X, G0Y, G0X2, G0Y2;

		cv::Mat dGXk = dGdXkernel(cv::Size(3,3),1);
		cv::Mat dGYk = dGdYkernel(cv::Size(3,3),1);

		cv::filter2D(sub1,G1X,-1,-dGXk);
		cv::filter2D(sub1,G1Y,-1,-dGYk);

		cv::filter2D(sub0,G0X,-1,-dGXk);
		cv::filter2D(sub0,G0Y,-1,-dGYk);

		arma::mat A(winSize.width*winSize.height,2);
		arma::mat B(winSize.width*winSize.height,1);

		arma::mat W(winSize.width*winSize.height,winSize.width*winSize.height);
		W.zeros();

		for(int xi =0 ; xi < winSize.width; xi++)
			for(int yi =0 ; yi < winSize.height; yi++)
			{
				A(yi+xi*winSize.height,0) = G1X.at<float>(yi,xi);
				A(yi+xi*winSize.height,1) = G1Y.at<float>(yi,xi);

//				B(yi+xi*winSize.height,0) = (float)dG.at<uchar>(yi,xi)*32;

				B(yi+xi*winSize.height,0) = (float)dG.at<float>(yi,xi);


//				float wx = (G1X.at<float>(yi,xi) - G0X.at<float>(yi,xi));
//				float wy = (G1Y.at<float>(yi,xi) - G0Y.at<float>(yi,xi));
//
//				float w = sqrt(wx*wx + wy*wy);
//				if(w !=0)
//					w = 1/w;
//				else
//					w = 100;
//
//				w = max(w,float(100.));

//				W(yi+xi*winSize.height,yi+xi*winSize.height) = w;
			}

		if(arma::det(A.t()*A)==0)
			break;

		arma::mat T = (A.t()*A).i()*A.t()*B;

		double bE = arma::norm(B);

		if(bE<=resErr)
		{
			resErr = bE;
			FtX = Ft1;
		}
		else
		{
			Ft1 = FtX;
			break;
		}

		cv::Point2f out(T(0,0),T(1,0));

		out += Ft1;

		Ft1 = out;

		arma::mat Tt = T.t()*T;

		if(Tt(0,0) < 1e-3)
			break;

		cout ++;
	}

//	std::cout<<"dp: "<<Ft1 - Ft0<<"\n";

	cv::Mat pl;
	img0.copyTo(pl);

	cv::cvtColor(pl,pl,CV_GRAY2BGR);

	cv::circle(pl,Ft1,5,cv::Scalar(0,255,255));

//	cv::imshow("img0",pl);
//	cv::waitKey(0);

	return Ft1;

}

cv::Mat imgFtRelated::pyramidConstructor(cv::Mat img0) {

	cv::Mat pryImg;

	int Sx = (img0.cols+1)/2;
	int Sy = (img0.rows+1)/2;

//	cv::Mat dst;
//
//	cv::GaussianBlur(img0,dst,cv::Size(3,3),sqrt(2),sqrt(2));
//	cv::resize(dst,pryImg,cv::Size(Sx,Sy));

	cv::Mat dst = cv::Mat::zeros(img0.rows/2,img0.cols/2,img0.type());

	for(int xi= 0; xi < img0.cols/2-1;xi++)
		for(int yi= 0; yi < img0.rows/2-1;yi++)
		{
			dst.at<uchar>(yi,xi) = 0.25*img0.at<uchar>(2*yi,2*xi) +
									0.125*img0.at<uchar>(2*yi,2*xi-1)+
									0.125*img0.at<uchar>(2*yi,2*xi+1)+
									0.125*img0.at<uchar>(2*yi-1,2*xi)+
									0.125*img0.at<uchar>(2*yi+1,2*xi)+
									0.0625*img0.at<uchar>(2*yi-1,2*xi-1)+
									0.0625*img0.at<uchar>(2*yi+1,2*xi+1)
											+0.0625*img0.at<uchar>(2*yi-1,2*xi+1)
											+0.0625*img0.at<uchar>(2*yi+1,2*xi-1);
		}
	dst.copyTo(pryImg);


	return pryImg;
}


cv::Point2f imgFtRelated::hmPryKLT(cv::Mat img0, cv::Mat img1, cv::Point2f Ft0,
		cv::Size winSize, int pryLvl) {

	std::vector<cv::Mat> pryImg0;
	std::vector<cv::Mat> pryImg1;

	std::vector<cv::Point2f> pryPt;

	cv::Mat imgL0;
	cv::Mat imgL1;

	img0.copyTo(imgL0);
	img1.copyTo(imgL1);

	cv::Point2f ptL = Ft0;

//// construct pry

	pryImg0.push_back(imgL0);
	pryImg1.push_back(imgL1);

	pryPt.push_back(Ft0);

	for(int i =0 ; i < pryLvl; i++)
	{
		cv::Mat imgL0p1 = pyramidConstructor(imgL0);
		cv::Mat imgL1p1 = pyramidConstructor(imgL1);

		imgL0 = imgL0p1;
		imgL1 = imgL1p1;

		pryImg0.push_back(imgL0);
		pryImg1.push_back(imgL1);

		cv::Point2f FtL(ptL.x/2,ptL.y/2);

		ptL = FtL;

		pryPt.push_back(FtL);
	}

//// pry KLT

	cv::Point2f ptK = pryPt[pryLvl-1];


	cv::Point2f dP(0,0);

	for(int i =0 ; i < pryLvl; i++)
	{
		int k = pryLvl-i-1;

		ptK = pryPt[k] + 2*dP;

//		ptK = pryPt[k] + dP;

		cv::Size pryWsize;
		pryWsize.width= winSize.width;//pow(1.41,k) ;
		pryWsize.height= winSize.height;//pow(1.41,k) ;

//		cv::Point2f ptKp1 = hmKLT(pryImg0[k], pryImg1[k],ptK, winSize);
//		cv::Point2f ptKp1 = hmCvKLT(pryImg0[k], pryImg1[k],ptK, pryWsize);

		cv::Point2f ptKp1 = hmCvWrapKLT(pryImg0[k], pryImg1[k],ptK, pryWsize);

//		cv::Point2f ptKp1 = hmAffineKLT(pryImg0[k], pryImg1[k],ptK, winSize);

		dP = (ptKp1 - ptK);

	}

	ptK = pryPt[0] + dP;

	return ptK;

}

bool imgFtRelated::harrisCornerCheck(cv::Point2f point, cv::Mat img) {

	double E = 0;

	int wSx = 50;
	int wSy = 50;

	float I0 = ((int)img.at<uchar>(point.y,point.x))/255.;

	cv::Mat M = cv::Mat::zeros(2,2,CV_32FC1);

	for(int xi = point.x - wSx; xi < point.x + wSx; xi++)
	{
		for(int yi = point.y - wSy; yi < point.y + wSy; yi++)
		{
			float I1 = ((int)img.at<uchar>(yi,xi))/255.;
			float Iy = ((int)img.at<uchar>(yi-1,xi))/255.;
			float Ix = ((int)img.at<uchar>(yi,xi-1))/255.;

			float dy = I1 - Iy;
			float dx = I1 - Ix;

			cv::Mat dM = cv::Mat::zeros(2,2,CV_32FC1);
			dM.at<float>(0,0) = dx*dx;
			dM.at<float>(1,1) = dy*dy;
			dM.at<float>(0,1) = dx*dy;
			dM.at<float>(1,0) = dx*dy;

			M+=dM;
		}
	}

	cv::Mat eigvalue;

	eigen(M,eigvalue);

//	std::cout<<eigvalue.at<float>(0,0)<<" "<<eigvalue.at<float>(1,0)<<"\n";

	E = (eigvalue.at<float>(0,0) - eigvalue.at<float>(1,0))/eigvalue.at<float>(0,0);

//	double detM = determinant(M);
//	double traceMa =  trace(Matx<double,2,2>(M));
//
//	float Ex = detM - traceMa;
//
//	std::cout<<E<<"\n";

//	if(E > 0.54 && E < 0.56)
	if(E > 0.4 && E < 1)
		return true;
	else
		return false;

}

void imgFtRelated::getFeatureGdtt(cv::Mat img, vector<cv::KeyPoint>& Ft) {

	double qualityLevel = 1e-5;
	double minDistance = 2;
	int blockSize = 4;
	bool useHarrisDetector = false;
	double k = 0.04;
	int maxCorners = 0;

	std::vector<cv::Point2f> PT;

	while(true)
	{
		std::vector<cv::Point2f> points;

		goodFeaturesToTrack(img,
				points,
				maxCorners,
				qualityLevel,
				minDistance,
				cv::Mat(),
				blockSize,
				useHarrisDetector,
				k );


		if(points.size() > 100)
		{
			minDistance *= 1.5;
		}
		else
		{
			PT = points;
			break;
		}
	}

	cv::KeyPoint::convert(PT,Ft);

}

cv::Point2f imgFtRelated::hmCvWrapKLT(cv::Mat img0, cv::Mat img1,
		cv::Point2f Ft0, cv::Size winSize) {

	cv::Point2f Ft1 = Ft0;

	int cout = 0;

	cv::Point2f C1(Ft0.x - winSize.width/2,Ft0.y - winSize.width/2);
	cv::Mat sub0;

	cv::Point2f Tr(0,0);
	//// check for boundary condition

	testWinSize(img0.size(),C1,winSize);

	sub0 = img0(cv::Rect(C1,winSize));
	sub0.convertTo(sub0,CV_32FC1,1./255,0);

	arma::mat dTduT(2,2);
	dTduT.zeros();

	cv::Mat warpMat = cv::Mat::zeros(2,3,CV_32FC1);
	warpMat.at<float>(0,0) = 1;
	warpMat.at<float>(1,1) = 1;

//	cv::Mat dGXk = dGdXkernel(cv::Size(5,5),1);
//	cv::Mat dGYk = dGdYkernel(cv::Size(5,5),1);

	cv::Mat dGXk = dGdXkernel(cv::Size(7,7),1);
	cv::Mat dGYk = dGdYkernel(cv::Size(7,7),1);

	cv::Mat G1X, G1Y,G1X2, G1Y2, G1Xf, G1Yf;
	cv::Mat G0X, G0Y, G0X2, G0Y2;

	cv::filter2D(sub0,G0X,-1,-dGXk);
	cv::filter2D(sub0,G0Y,-1,-dGYk);

	double errdG = 100;

	while(cout < 50)
	{
		warpMat.at<float>(0,2) = Ft1.x - winSize.width/2;
		warpMat.at<float>(1,2) = Ft1.y - winSize.height/2;

		cv::Mat sub1;

		cv::warpAffine(img1,sub1,warpMat,winSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP);

		sub1.convertTo(sub1,CV_32FC1,1./255,0);

		cv::Mat dG = sub0 - sub1;

//		cv::imshow("0",sub0);
//		cv::imshow("1",sub1);
//		cv::imshow("2",dG);
//		cv::waitKey(0);

		cv::Mat S0, S1, dS;

		cv::filter2D(sub1,G1X,-1,-dGXk);
		cv::filter2D(sub1,G1Y,-1,-dGYk);

//		cv::warpAffine(G1Xf,G1X,warpMat,winSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP);
//		cv::warpAffine(G1Yf,G1Y,warpMat,winSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP);

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

				B(0,0) += (float)dG.at<float>(yi,xi)*G1X.at<float>(yi,xi);
				B(1,0) += (float)dG.at<float>(yi,xi)*G1Y.at<float>(yi,xi);
			}

		A(1,0) = A(0,1);

		if(arma::det(A.t()*A)==0)
			break;

		arma::mat T = (A).i()*B;

//		T.t().print("dT");

		cv::Point2f out(T(0,0),T(1,0));

		out += Ft1;

		Ft1 = out;

		if(fabs(T(0,0)) < 1e-2 && fabs(T(1,0)) < 1e-2)
			break;

		cout ++;
	}

	return Ft1;

}
