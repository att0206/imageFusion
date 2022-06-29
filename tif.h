#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>
#include<omp.h>
#include <xmmintrin.h>

#define ENABLE false

#define triple(x,up,down) ((x) > (up) ? (up):((x) < (down)) ? (down) : ((uchar)(x)))
class tif
{
public:
	tif(int rows,int cols,int M);
	~tif();
	float* imfilter(const float* in, int rows, int cols,const int M);
	float* arraySub(const float* in, const float* in2, int length);
	float* arrayPow(const float* in, int length, int value);
	float* arraySubAndPow(const float* in, const float* in2, int length, int value);
	float* arraySubAndPowSSE(const float* in,const float* in2,int length,int value);
	float* integralImage(float* in,int rows,int cols);
	float* imfilterInteImage(float* in,int rows,int cols,int M);
	cv::Mat run(cv::Mat t1, cv::Mat t2);
	inline uchar Triple(float x,uchar up,uchar down);
private:
	int rows;
	int cols;
	int m_length;
	int m_M;
	float* t1PadPtr;
	float* t2PadPtr;
	float* t1Ptr;
	float* t2Ptr;
	float* Med1;
	float* Med2;
	float* w1;
	float* w2;
	float* F1;
	float* F2;

	//float* sseOut;
	cv::Mat FF;
};

