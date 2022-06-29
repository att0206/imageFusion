#include "tif.h"

tif::tif(int rows, int cols, int M) {
	rows = rows;
	cols = cols;
	m_M = M;
	int pad = int(M / 2);
	m_length = rows * cols;
	int padLength = (cols + 2 * pad) * (rows + 2 * pad);
	t1PadPtr = new float[padLength];
	t2PadPtr = new float[padLength];
	t1Ptr = new float[m_length];
	t2Ptr = new float[m_length];
	Med1 = new float[m_length];
	Med2 = new float[m_length];
	w1 = new float[m_length];
	w2 = new float[m_length];
	F1 = new float[m_length];
	F2 = new float[m_length];
	FF = cv::Mat::zeros(rows,cols,CV_8UC1);
#if ENABLE
	omp_set_num_threads(omp_get_max_threads());
#endif
}
tif::~tif() {
	if (t1PadPtr)
		delete[] t1PadPtr;
	if (t2PadPtr)
		delete[] t2PadPtr;
	if (t1Ptr)
		delete[] t1Ptr;
	if (t2Ptr)
		delete[] t2Ptr;
	if (Med1)
		delete[] Med1;
	if (Med2)
		delete[] Med2;
	if (w1)
		delete[] w1;
	if (w2)
		delete[] w2;
	if (F1)
		delete[] F1;
	if (F2)
		delete[] F2;
}
inline uchar tif::Triple(float x, uchar up, uchar down) {
	uchar value = x > up ? up : (x < 0 ? 0 : (uchar)x);
	return value;
}
float* tif::imfilter(const float* in, int rows, int cols,const int M) {//需要进一步优化 卷积计算
	if (in == nullptr)
		return nullptr;
	int pad = int(M / 2);
	int length = (rows - 2 * pad) * (cols - 2 * pad);
	float* out = new float[length];
	float scale = 1.0 / (M * M);
#if(ENABLE)
#pragma omp parallel for
#endif
	for (auto iii = pad; iii < rows - pad; iii++) {//基本全部耗时都是在这个4层for循环中  需要进一步优化
		for (auto jjj = pad; jjj < cols - pad; jjj++) {
			float filterSum = 0.0;
			for (auto offsetX = -pad; offsetX < pad + 1; offsetX++) {
				int indexX = iii + offsetX;
				for (auto offsetY = -pad; offsetY < pad + 1; offsetY++) {
					int indexY = jjj + offsetY;
					int pos = (indexX * cols) + indexY;
					filterSum += scale * in[pos];
				}
			}
			out[(iii - pad) * (cols - 2 * pad) + jjj - pad] = filterSum;
		}
	}
	return out;
}

float* tif::arraySub(const float* in, const float* in2, int length) {
	if (in == nullptr)
		return nullptr;
	float* out = new float[length];
	//memcpy(out,0.,sizeof(float)*length);
	int iii;
	int circle = length / 8;
#if(ENABLE)
#pragma omp parallel for
#endif
	for (iii = 0; iii < length-8; iii+=8) {
		//out[iii] = in[iii] - in2[iii];
		__m256 inS = _mm256_load_ps(&in[iii]);
		__m256 in2S = _mm256_load_ps(&in2[iii]);
		__m256 outS = _mm256_sub_ps(inS,in2S);
		_mm256_store_ps(&out[iii],outS);
		//std::cout <<iii<<" is "<< omp_get_thread_num() << std::endl;
	}
	for (iii = 8*circle; iii < length;iii++) {
		out[iii] = in[iii] - in2[iii];
	}
	return out;
}

float* tif::arrayPow(const float* in, int length, int value) {

	if (in == nullptr)
		return nullptr;
	float* out = new float[length];
	int iii;
	int circle = length / 8;
#if(ENABLE)
#pragma omp parallel for
#endif
	for (iii = 0; iii < length - 8; iii+=8) {
		//float data = in[iii];
		__m256 temp = _mm256_load_ps(&in[iii]);
		//out[iii] = data;
		__m256 dotReult = temp;
		//for (auto powValue = 0; powValue < value; powValue++) {
			//out[iii] *= data;
			dotReult = _mm256_mul_ps(dotReult,temp);
		//}
		_mm256_store_ps(&out[iii],dotReult);
	}

	for (iii = 8*circle; iii < length;iii++) {
		float data = in[iii];
		out[iii] = data;
		for (auto powValue = 0; powValue < value - 1;powValue++) {
			out[iii] *= data;
		}
	}
	return out;
}
float* tif::arraySubAndPow(const float* in, const float* in2, int length, int value) {

	if (in == nullptr || in2 == nullptr)
		return nullptr;
	float* out = new float[length];
	//memcpy(out,0.,sizeof(float)*length);
#if(ENABLE)
#pragma omp parallel for
#endif
	for (auto iii = 0; iii < length; iii++) {
		float data = in[iii] - in2[iii];
		//float data = out[iii];
		out[iii] = data;
		for (auto powValue = 0; powValue < value-1; powValue++) {
			out[iii] *= data;
		}
	}
	return out;
}

float* tif::arraySubAndPowSSE(const float* in, const float* in2, int length, int value) {
	
	if (in == nullptr || in2 == nullptr)
		return nullptr;
	float* out = new float[length];
	int circle = length / 8;
	int iii;
#if(ENABLE)
#pragma omp parallel for
#endif
	for (iii = 0; iii < length - 8; iii+=8) {
		__m256 inS = _mm256_load_ps(&in[iii]);
		__m256 in2S = _mm256_load_ps(&in2[iii]);
		__m256 subResult = _mm256_sub_ps(inS,in2S);
		__m256 powResult = _mm256_mul_ps(subResult,subResult);
		_mm256_store_ps(&out[iii],powResult);
	}

	for (iii = 8*circle; iii < length;iii++) {
		float sub = in[iii] - in2[iii];
		out[iii] = sub * sub;
	}
	return out;
}

float* tif::integralImage(float* in, int rows, int cols) {
	if (in == nullptr)
		return nullptr;
	int length = rows * cols;
	float* out = new float[length];
	out[0] = in[0];
	for (auto iii = 1; iii < cols;iii++) {
		out[iii] = in[iii] + out[iii - 1];
	}

	for (auto iii = 1; iii < rows;iii++) {
		float colsLocSum = 0;
		int jjj;
		for (jjj = 0; jjj < cols - 4;jjj+=4) {
			int index = iii * cols + jjj;
			int befLine = (iii - 1) * cols + jjj;
			float num1 = in[index] + in[index + 1];
			float num2 = in[index + 2] + num1;
			float num3 = in[index + 3] + num2;
			__m128 sum_sse = _mm_add_ps(_mm_set1_ps(colsLocSum),_mm_set_ps(num3,num2,num1,in[index]));
			__m128 temp = _mm_add_ps(_mm_set_ps(out[befLine + 3],out[befLine + 2],out[befLine + 1],out[befLine]),sum_sse);
			_mm_store_ps(&out[index],temp);
			colsLocSum += num3;
		}
		for (; jjj < cols;jjj++) {
			int index = iii * cols + jjj;
			colsLocSum += in[index];
			out[index] = colsLocSum + out[(iii - 1) * cols + jjj];
		}
	}
	return out;
}

float* tif::imfilterInteImage(float* in, int rows, int cols, int M) {
	if (in == nullptr)
		return nullptr;
	float* inteImageOutPtr = integralImage(in,rows,cols);
	const int pad = M / 2;
	const int length = (rows - 2 * pad) * (cols - 2 * pad);
	float* out = new float[length];
	float scale = 1.0 / (M*M);
	for (auto iii = pad; iii < rows - pad;iii++) {
		for (auto jjj = pad; jjj < cols - pad;jjj++) {
			int outIndex = (iii - pad) * (cols - 2 * pad) + jjj - pad;
			if (iii == pad) {
				out[outIndex] = inteImageOutPtr[(iii + pad) * cols + jjj + pad] - inteImageOutPtr[(iii - 1) * cols + jjj + pad];
				continue;
			}
			if (jjj == pad) {
				out[outIndex] = inteImageOutPtr[(iii + pad) * cols + jjj + pad] - inteImageOutPtr[(iii + pad) * cols + jjj - 1];
				continue;
			}
			out[outIndex] = (inteImageOutPtr[(iii + pad) * cols + jjj + pad] - inteImageOutPtr[(iii - pad - 1) * cols + jjj + pad] - inteImageOutPtr[(iii + pad) * cols + jjj - pad - 1] + inteImageOutPtr[(iii - pad - 1)*cols + jjj - pad - 1])*scale;
		}
	}
	if (inteImageOutPtr)
		delete[] inteImageOutPtr;
	return out;
}
cv::Mat tif::run(cv::Mat t1, cv::Mat t2) {
	cv::Mat med1, med2;
	cv::medianBlur(t1, med1, 3);
	cv::medianBlur(t2, med2, 3);
	cv::Mat t1pad, t2pad;
	int pad = int(m_M / 2);
	cv::copyMakeBorder(t1, t1pad, pad, pad, pad, pad, cv::BORDER_CONSTANT);
	cv::copyMakeBorder(t2, t2pad, pad, pad, pad, pad, cv::BORDER_CONSTANT);
#if(ENABLE)
#pragma omp parallel for
#endif
	for (auto iii = 0; iii < t1.rows * t1.cols; iii++) {
		t1Ptr[iii] = t1.data[iii];
		t2Ptr[iii] = t2.data[iii];
		Med1[iii] = med1.data[iii];
		Med2[iii] = med2.data[iii];
	}
#if(ENABLE)
#pragma omp parallel for
#endif
	for (auto iii = 0; iii < t1pad.rows * t1pad.cols; iii++) {
		t1PadPtr[iii] = t1pad.data[iii];
		t2PadPtr[iii] = t2pad.data[iii];
	}
	double time1 = (double)cv::getTickCount();
	//float* b1 = imfilter(t1PadPtr, t1pad.rows, t1pad.cols, m_M);
	float* b1 = imfilterInteImage(t1PadPtr, t1pad.rows, t1pad.cols, m_M);
	std::cout <<"imfilter :"<< ((double)cv::getTickCount() - time1) / cv::getTickFrequency() << std::endl;
	float* d1 = arraySub(t1Ptr, b1, t1.rows * t1.cols);
	//float* b1Med1 = arraySub(b1, Med1, t1.rows * t1.cols);
	//float* S1 = arrayPow(b1Med1, t1.rows * t1.cols, 2);
	float* S1 = arraySubAndPow(b1,Med1, t1.rows * t1.cols,2);
	//float* b2 = imfilter(t2PadPtr, t2pad.rows, t2pad.cols, m_M);
	float* b2 = imfilterInteImage(t2PadPtr, t2pad.rows, t2pad.cols, m_M);
	float* d2 = arraySub(t2Ptr, b2, t2.rows * t2.cols);
	//float* b2Med2 = arraySub(b2, Med2, t2.rows * t2.cols);
	//float* S2 = arrayPow(b2Med2, t2.rows * t2.cols, 2);
	float* S2 = arraySubAndPow(b2, Med2, t2.rows * t2.cols, 2);
	int num = m_length / 8;
#if(ENABLE)
#pragma omp parallel for
#endif
	for (auto iii = 0; iii < m_length - 8; iii+=8) {
		__m256 S1S = _mm256_load_ps(&S1[iii]);
		__m256 S2S = _mm256_load_ps(&S2[iii]);
		__m256 w1S = _mm256_div_ps(S1S, _mm256_add_ps(S1S,S2S));
		__m256 w2S = _mm256_div_ps(S2S, _mm256_add_ps(S1S, S2S));
		_mm256_store_ps(&w1[iii], w1S);
		_mm256_store_ps(&w2[iii], w2S);
	}
	for (auto iii = 8 * num; iii < m_length;iii++) {
		w1[iii] = S1[iii] / (S1[iii] + S2[iii]);
		w2[iii] = S2[iii] / (S1[iii] + S2[iii]);
	}
	float value = 0.0;
	__m256 param = _mm256_set1_ps(0.5f);
	int iii;
	int circle = m_length / 8;
#if(ENABLE)
#pragma omp parallel for
#endif
	for (iii = 0; iii < m_length - 8; iii+=8) {
		__m256 w1S = _mm256_load_ps(&w1[iii]);
		__m256 d1S = _mm256_load_ps(&d1[iii]);
		__m256 w2S = _mm256_load_ps(&w2[iii]);
		__m256 d2S = _mm256_load_ps(&d2[iii]);
		__m256 b1S = _mm256_load_ps(&b1[iii]);
		__m256 b2S = _mm256_load_ps(&b2[iii]);
		__m256 F1S = _mm256_add_ps(_mm256_mul_ps(w1S,d1S),_mm256_mul_ps(w2S,d2S));
		__m256 F2S = _mm256_add_ps(_mm256_mul_ps(param,b1S),_mm256_mul_ps(param,b2S));
		__m256 valueS = _mm256_add_ps(F1S,F2S);
		_mm256_store_ps(&S2[iii],valueS);
	}
	for (iii = 8*circle; iii < m_length;iii++) {
		F1[iii] = w1[iii] * d1[iii] + w2[iii] * d2[iii];
		F2[iii] = 0.5 * b1[iii] + 0.5 * b2[iii];
		S2[iii] = F1[iii] + F2[iii];	 
	}
	for (iii = 0; iii < m_length;iii++) {
		FF.data[iii] = Triple(S2[iii],255,0);
	}

	delete[] b1;
	delete[] d1;
	delete[] S1;
	delete[] b2;
	delete[] d2;
	delete[] S2;
	return FF;
}