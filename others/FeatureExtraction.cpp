
//#include "stdafx.h" 
#include <iostream>  
#include <fstream>  
#include<algorithm>
#include <sstream>
#include <stdio.h>
#include <opencv.hpp>
#include <core/core.hpp>   
#include <highgui/highgui.hpp>   
#include <features2d/features2d.hpp>  
#include <windows.h> 
#include<math.h>
#include<io.h>
#include <string>
#include <fstream>
#include <sstream>
#include <direct.h>  

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <Windows.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
//#include "opencv2/xfeatures2d.hpp"
#include <windows.h>
#include <tchar.h> 
#include <stdio.h>
#include <strsafe.h>

#pragma comment(lib, "User32.lib")


using namespace cv;
using namespace std;

double RansacThreshold = 2;

vector<string> ReadIm()
{
	WIN32_FIND_DATA ffd;
	LARGE_INTEGER filesize;
	TCHAR szDir[MAX_PATH];
	size_t length_of_arg;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	vector<string> files;
	StringCchLength(".", MAX_PATH, &length_of_arg);


	StringCchCopy(szDir, MAX_PATH, ".");
	StringCchCat(szDir, MAX_PATH, TEXT("\\data\\*"));
	hFind = FindFirstFile(szDir, &ffd);


	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			//_tprintf(TEXT("  %s   <DIR>\n"), ffd.cFileName);
		}
		else
		{
			string t = "./data/";
			filesize.LowPart = ffd.nFileSizeLow;
			filesize.HighPart = ffd.nFileSizeHigh;
			//_tprintf(TEXT("  %s   %ld bytes\n"), ffd.cFileName, filesize.QuadPart);
			files.push_back(t.append(ffd.cFileName));
		}
	} while (FindNextFile(hFind, &ffd) != 0);

	FindClose(hFind);
	return files;
}



class Tracking
{
public:
	void ExtractORBFeature(Mat img, vector<Point2f> & points)
	{
		vector<KeyPoint> keypoints;
		Ptr<ORB> orb = ORB::create(1000, 2, 4, 5, 0, 2, ORB::HARRIS_SCORE, 5);
		orb->detect(img, keypoints);
		for (int i = 0; i < keypoints.size(); i++)
		{
			points.push_back(keypoints[i].pt);
		}
	}
	void LKCorrespondence(Mat img_1, Mat img_2, vector<Point2f> & points0, vector<Point2f> & points1)
	{
		Size subPixWinSize(10, 10), winSize(5, 5);
		TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 15, 0.1);
		vector<uchar> status;
		vector<float> err;
		calcOpticalFlowPyrLK(img_1, img_2, points0, points1, status, err, winSize,
			3, termcrit);
		for (int i = points1.size() - 1; i >= 0; i--)
		{
			if (status[i] == 0)
			{
				points1.erase(points1.begin() + i, points1.begin() + i + 1);
				points0.erase(points0.begin() + i, points0.begin() + i + 1);
			}
		}
	}
	void Match(Mat img_1, Mat img_2, vector<Point2f> & points0, vector<Point2f> & points1)
	{
		BFMatcher matcher(NORM_HAMMING);
		Ptr<ORB> orb = ORB::create(1000, 2, 4, 5, 0, 2, ORB::HARRIS_SCORE, 5);
		vector<Mat> descriptors;
		vector<KeyPoint> keypoints_1, keypoints_2;
		Mat descriptors_1, descriptors_2;
		//orb->detect(img_1, keypoints_1);
		//orb->detect(img_2, keypoints_2);
		for (size_t i = 0; i < points0.size(); i++) {
			keypoints_1.push_back(cv::KeyPoint(points0[i], 1.f));
		}
		for (size_t i = 0; i < points1.size(); i++) {
			keypoints_2.push_back(cv::KeyPoint(points1[i], 1.f));
		}
		orb->compute(img_1, keypoints_1, descriptors_1);
		orb->compute(img_2, keypoints_2, descriptors_2);
		descriptors.push_back(descriptors_1);
		matcher.add(descriptors);
		vector<vector<DMatch>> matches, goodmatches;
		DMatch bestMatch, betterMatch;
		vector<DMatch> bestMatches;
		matcher.knnMatch(descriptors_2, matches, 2);
		int n = 0;
		Point p1, p2;
		points0.clear();
		points1.clear();
		for (int i = 0; i<(int)matches.size(); i++)
		{
			bestMatch = matches[i][0];
			betterMatch = matches[i][1];
			p1 = keypoints_1[bestMatch.trainIdx].pt;
			p2 = keypoints_2[bestMatch.queryIdx].pt;
			double distance = sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
			float distanceRatio = bestMatch.distance / betterMatch.distance;

			if (distanceRatio< 0.8 && distance<30)
			{
				bestMatches.push_back(bestMatch);
				points0.push_back(p1);
				points1.push_back(p2);
				//line(img_2, p1, p2, Scalar(0, 0, 255), 1, 8, 0);
			}
		}
	}
	Mat Triangulate(vector<Point2f> & points0, vector<Point2f> & points1)
	{
		vector <uchar> RANSACStatus;
		Mat fundamental_matrix =
			findFundamentalMat(points0, points1, FM_RANSAC, RansacThreshold, 0.99, RANSACStatus);
		for (int i = points1.size() - 1; i >= 0; i--)
		{
			if (RANSACStatus[i] == 0)
			{
				points1.erase(points1.begin() + i, points1.begin() + i + 1);
				points0.erase(points0.begin() + i, points0.begin() + i + 1);
			}
		}
		return fundamental_matrix;
	}
	Mat defineIntrinsic()
	{
		double fx, fy, cx, cy;
		Mat K = cv::Mat::eye(3, 3, CV_64FC1);
		/*fx = 517.3;
		fy = 516.5;
		cx = 318.6;
		cy = 255.3;*/
		//For Iphone Image
		fx = 519.97454;
		fy = 521.06369;
		cx = 236.80906;
		cy = 320.6;
		K.at<double>(0, 0) = fx;
		K.at<double>(1, 1) = fy;
		K.at<double>(0, 2) = cx;
		K.at<double>(1, 2) = cy;
		return K;
	}
	Mat MatrixToQuaternion(Mat R)
	{
		Mat q(4, 1, CV_64FC1);
		q.at<double>(0, 0) = sqrt(1 + R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2)) / 2;
		q.at<double>(1, 0) = (R.at<double>(1, 2) - R.at<double>(2, 1)) / (4 * q.at<double>(0, 0));
		q.at<double>(2, 0) = (R.at<double>(2, 0) - R.at<double>(0, 2)) / (4 * q.at<double>(0, 0));
		q.at<double>(3, 0) = (R.at<double>(0, 1) - R.at<double>(1, 0)) / (4 * q.at<double>(0, 0));
		return q;
	}
	Mat QuaternionToMatrix(Mat q)
	{
		Mat R(3, 3, CV_64FC1);
		R.at<double>(0, 0) = 1 - 2 * pow(q.at<double>(2, 0), 2) - 2 * pow(q.at<double>(3, 0), 2);
		R.at<double>(0, 1) = 2 * q.at<double>(1, 0)*q.at<double>(2, 0) + 2 * q.at<double>(0, 0)*q.at<double>(3, 0);
		R.at<double>(0, 2) = 2 * q.at<double>(1, 0)*q.at<double>(3, 0) - 2 * q.at<double>(0, 0)*q.at<double>(2, 0);
		R.at<double>(1, 0) = 2 * q.at<double>(1, 0)*q.at<double>(2, 0) - 2 * q.at<double>(0, 0)*q.at<double>(3, 0);
		R.at<double>(1, 1) = 1 - 2 * pow(q.at<double>(1, 0), 2) - 2 * pow(q.at<double>(3, 0), 2);
		R.at<double>(1, 2) = 2 * q.at<double>(2, 0)*q.at<double>(3, 0) + 2 * q.at<double>(0, 0)*q.at<double>(1, 0);
		R.at<double>(2, 0) = 2 * q.at<double>(1, 0)*q.at<double>(3, 0) + 2 * q.at<double>(0, 0)*q.at<double>(2, 0);
		R.at<double>(2, 1) = 2 * q.at<double>(2, 0)*q.at<double>(3, 0) - 2 * q.at<double>(0, 0)*q.at<double>(1, 0);
		R.at<double>(2, 2) = 1 - 2 * pow(q.at<double>(1, 0), 2) - 2 * pow(q.at<double>(2, 0), 2);
		return R;
	}
	
};
int main(int argc, char** argv)
{
	vector <string> files=ReadIm();
	int interval = 1;
	int i = 0;
	vector<Mat> Orientations, poss, Rs;
	
	Mat BenchMarkFrame= imread(files[i]);

	//For Iphone Image
	//Rect Template(161, 0, 960, 720);
	//imshow("0", BenchMarkFrame);
	//CurFrame = CurFrame(Template);
	resize(BenchMarkFrame, BenchMarkFrame, Size(640, 480));
	//Mat gray;
	//cvtColor(BenchMarkFrame, gray, COLOR_BGR2GRAY);
	//imshow("1", gray);
	//waitKey(0);


	Mat PreFrame, CurFrame;
	vector<Point2f> BenchMarkPoints;
	Ptr<Tracking> tracker;
	tracker->ExtractORBFeature(BenchMarkFrame, BenchMarkPoints);
	vector<Point2f> CurPoints(BenchMarkPoints.size()), PrePoints;
	copy(BenchMarkPoints.begin(), BenchMarkPoints.end(), CurPoints.begin());
	BenchMarkFrame.copyTo(CurFrame);

	Mat K = tracker->defineIntrinsic();
	Mat Kt = K.t();
	i += interval;
	double focal = 517;
	Mat CurOrientation(4, 1, CV_64FC1), PreOrientation(4, 1, CV_64FC1), Curpos(3, 1, CV_64FC1), Prepos(3, 1, CV_64FC1);
	Mat CurR(3, 3, CV_64FC1), PreR(3, 3, CV_64FC1), CurT(3, 1, CV_64FC1), PreT(3, 1, CV_64FC1);
	//Initialization For RGB Dataset
	CurOrientation.at<double>(0, 0) = 0.8596;
	CurOrientation.at<double>(1, 0) = -0.3534;
	CurOrientation.at<double>(2, 0) = 0.0838;
	CurOrientation.at<double>(3, 0) = -0.3594;
	Curpos.at<double>(0, 0) = 0.4388;
	Curpos.at<double>(1, 0) = -0.4332;
	Curpos.at<double>(2, 0) = 1.4779;

	//Initialization For Iphone


	CurR = tracker->QuaternionToMatrix(CurOrientation).t();
	CurT = -1 * CurR*Curpos;


	Orientations.push_back(CurOrientation);
	//Rs.push_back(CurR);
	Mat tempt(1, 3, CV_64FC1);
	tempt = Curpos.clone();
	poss.push_back(tempt);
	//poss.push_back(Curpos);
	//int q = 1;
	int BenchMarkFlag = 1;

	while (i < files.size())
	{
		CurFrame.copyTo(PreFrame);
		CurFrame = imread(files[i]);
		//imshow("0", CurFrame);
		//For Iphone Image
		//Rect Template(161, 0, 960, 720);
		//CurFrame = CurFrame(Template);
		resize(CurFrame, CurFrame, Size(640, 480));
		//imshow("1",CurFrame);

		PrePoints.clear();
		PrePoints.resize(CurPoints.size());
		copy(CurPoints.begin(), CurPoints.end(), PrePoints.begin());
		//Check if CurPoints could have a size nonzero

		tracker->LKCorrespondence(PreFrame, CurFrame, PrePoints, CurPoints);

		////Use Match
		//CurPoints.clear();
		////CurFrame.copyTo(BenchMarkFrame);
		//tracker->ExtractORBFeature(CurFrame, CurPoints);
		//tracker->Match(PreFrame, CurFrame, PrePoints, CurPoints);

		
		// Show the match between two adjacent frames
		/*Mat ShowMatches;
		CurFrame.copyTo(ShowMatches);
		for (int i = 0; i < PrePoints.size(); i++)
		{
			line(ShowMatches, PrePoints[i], CurPoints[i], Scalar(0, 0, 255), 1, 8, 0);
		}
		imshow("MatchesBetweenAdjacentFrames",ShowMatches);
		waitKey(0);*/



		if (BenchMarkFlag == 1)
		{
			BenchMarkPoints.clear();
			BenchMarkPoints.resize(PrePoints.size());
			copy(PrePoints.begin(), PrePoints.end(), BenchMarkPoints.begin());
			BenchMarkFlag = 0;
			cout << i << endl;
		}


		Mat fundamental_matrix = tracker->Triangulate(PrePoints, CurPoints);
		// Show the match between two adjacent frames
		//Mat ShowMatches;
		//CurFrame.copyTo(ShowMatches);
		//for (int i = 0; i < PrePoints.size(); i++)
		//{
		//line(ShowMatches, PrePoints[i], CurPoints[i], Scalar(0, 0, 255), 1, 8, 0);
		//}
		//imshow("MatchesBetweenAdjacentFrames",ShowMatches);
		//waitKey(0);


		//Show Epipolar Lines:
		if (1)
		{
			std::vector <cv::Vec3f> lines1;
			computeCorrespondEpilines(PrePoints, 1, fundamental_matrix, lines1);
			Mat Show;
			CurFrame.copyTo(Show);
			for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it)
			{
				// Draw the line between first and last column
				cv::line(Show,
					cv::Point(0, -(*it)[2] / (*it)[1]),
					cv::Point(Show.cols, -((*it)[2] +
					(*it)[0] * Show.cols) / (*it)[1]),
					cv::Scalar(0, 255, 0));
			}
			//imshow("Epipolarline", Show);
			imwrite(to_string(i)+".jpg", Show);
			//waitKey(0);
		}
		
		Mat E = Kt*fundamental_matrix*K;
		Mat TemptR, TemptT, Ori_, Pos_relative;
		Curpos.copyTo(Prepos);
		CurOrientation.copyTo(PreOrientation);
		CurR.copyTo(PreR);
		CurT.copyTo(PreT);
		recoverPose(E, PrePoints, CurPoints, TemptR, TemptT, focal);
		//If the outputs are the orientation and the position.
		//TemptR = TemptR.t();
		//TemptT = -1 * TemptR*TemptT;


		CurR = TemptR*PreR;
		CurOrientation = tracker->MatrixToQuaternion(CurR.t());
		Curpos = -1 * tracker->QuaternionToMatrix(PreOrientation)*TemptR.inv()*TemptT + Prepos;
		CurT =-1* CurR*Curpos;
		/*cout << Curpos  << endl;*/
		//CurT = TemptT + PreT;
		//Curpos = -1 * CurR.inv()*CurT;

		double PointsRatio = double(CurPoints.size()) / double(BenchMarkPoints.size());
		if (PointsRatio < 0.6|| CurPoints.size()<80)
		{
			//size changes
			BenchMarkPoints.clear();
			CurFrame.copyTo(BenchMarkFrame);
			tracker->ExtractORBFeature(BenchMarkFrame, BenchMarkPoints);
			CurPoints.clear();
			CurPoints.resize(BenchMarkPoints.size());
			copy(BenchMarkPoints.begin(), BenchMarkPoints.end(), CurPoints.begin());
			BenchMarkFlag = 1;
		}
		Orientations.push_back(CurOrientation);
		Mat tempt(1, 3, CV_64FC1);
		tempt=Curpos.clone();
		poss.push_back(tempt);
		//std::cout<< poss.back().at<double>(0, 0) << " " << poss.back().at<double>(1, 0) << " " << poss.back().at<double>(2, 0) << " "<<endl;
		//cout << poss.back() << endl;
		//std::cout << poss[0].at<double>(0, 0) << " " << poss[0].at<double>(1, 0) << " " << poss[0].at<double>(2, 0) << " " << endl;
		//q++;
		i += interval;
	}

	ofstream location_out;
	string filename;
	filename = "Interval=" + to_string(interval) + "_Ransac=" + to_string(RansacThreshold) + ".txt";
	location_out.open(filename, std::ios::out | std::ios::app);  
	if (!location_out.is_open())
		return 0;
	int num = Orientations.size();
	location_out << num << endl;
	for (int j = 0; j < num; j++)
	{
		Mat Tempt = tracker->QuaternionToMatrix(Orientations[j]);
		location_out << K.at<double>(0, 0) << " " << K.at<double>(0, 1) << " " << K.at<double>(0, 2) << " " << K.at<double>(1, 0) << " " << K.at<double>(1, 1) << " " << K.at<double>(1, 2) << " " << K.at<double>(2, 0) << " " << K.at<double>(2, 1) << " " << K.at<double>(2, 2) << " ";
		location_out << Tempt.at<double>(0, 0) << " " << Tempt.at<double>(0, 1) << " " << Tempt.at<double>(0, 2) << " " << Tempt.at<double>(1, 0) << " " << Tempt.at<double>(1, 1) << " " << Tempt.at<double>(1, 2) << " " << Tempt.at<double>(2, 0) << " " << Tempt.at<double>(2, 1) << " " << Tempt.at<double>(2, 2) << " ";
		location_out << poss[j].at<double>(0, 0) << " " << poss[j].at<double>(1, 0) << " " << poss[j].at<double>(2, 0) << " ";
		/*cout << poss[0].at<double>(0, 0) << " " << poss[0].at<double>(1, 0) << " " << poss[0].at<double>(2, 0) << " " << endl;*/
		//cout << poss[1].at<double>(0, 0) << " " << poss[1].at<double>(1, 0) << " " << poss[1].at<double>(2, 0) << " " << endl;
		//cout << poss[2].at<double>(0, 0) << " " << poss[2].at<double>(1, 0) << " " << poss[2].at<double>(2, 0) << " " << endl;
		//cout << poss[3].at<double>(0, 0) << " " << poss[3].at<double>(1, 0) << " " << poss[3].at<double>(2, 0) << " " << endl;
		location_out << 640 << " " << 480 << endl;
	}

	location_out.close();
	//
	//Mat img_1 = imread("1305031790.645155.png");
	//Mat img_2 = imread("1305031790.713097.png");
	//namedWindow("Matches");
	//DWORD t1, t2,t0,t3;
	//t1 = GetTickCount();

	//
	//t0 = GetTickCount();
	//vector<Point2f> points[2];

	//tracker->ExtractORBFeature(img_1, points[0]);
	////Mat gray;
	////cvtColor(img_1, gray, COLOR_BGR2GRAY);
	////goodFeaturesToTrack(gray, points[0], 1000, 0.01, 10, Mat(), 3, 3, 0, 0.04);
	////int t = keypoints_1.size();



	//t3 = GetTickCount();
	///*Mat gray1, gray2;
	//cvtColor(img_1, gray1, COLOR_BGR2GRAY);
	//cvtColor(img_2, gray2, COLOR_BGR2GRAY);*/
	//tracker->LKCorrespondence(img_1, img_2, points[0], points[1]);
	//Mat fundamental_matrix = tracker->Triangulate(points[0], points[1]);
	//Mat K = tracker->defineIntrinsic();
	//Mat Kt = K.t();
	//Mat E = Kt*fundamental_matrix*K;
	//Mat R,t;
	//double focal = 517;
	//recoverPose(E, points[0], points[1], R, t, focal);
	//cout << "R=" << R <<"\n" << "t=" << t << endl;
	//Mat q = tracker->MatrixToQuaternion(R);
	//cout << "q=" << q << endl;
	////sort(err.begin(),err.end());
	//for (int i = 0; i < points[0].size(); i++)
	//{
	//	line(img_2, points[0][i], points[1][i], Scalar(0, 0, 255), 1, 8, 0);
	//}
	////cout << "F=" << fundamental_matrix << endl;
	//

	//Mat q1(4, 1, CV_64FC1);
	//q1.at<double>(0, 0) = 0.8596; 
	//q1.at<double>(1, 0) = -0.3534;
	//q1.at<double>(2, 0) = 0.0838;
	//q1.at<double>(3, 0) = -0.3594;
	//Mat R1 = tracker->QuaternionToMatrix(q1);
	//cout << "R1=" << R1 << endl;
	//Mat Rrel = R.t()*R1;
	//Mat q2 = tracker->MatrixToQuaternion(Rrel);
	//Mat q3 = tracker->MatrixToQuaternion(R1);
	//cout << "q2=" << q2 << endl;
	//cout << "=" << q3 << endl;
	//t2 = GetTickCount();
	//imshow("Matches", img_2);
	//cout << t2 - t1 << "ms"<< '\n' << t0 - t1 << "ms"<< '\n' << t3 - t0 << "ms"<<'\n'<<t2-t3<<"ms";




	//waitKey(0);
	return 0;
}