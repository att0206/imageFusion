#include<opencv2/opencv.hpp>
#include"tif.h"

int main(int argc,char** argv) {

	std::string kjgPath = "C:\\workspace\\2022\\imagefusion\\input\\VI\\";
	std::string infraPath = "C:\\workspace\\2022\\imagefusion\\input\\IR\\";
	std::string name = "snow.jpg";
	cv::Mat kjg = cv::imread(kjgPath + name, 0);
	cv::Mat infra = cv::imread(infraPath + name,0);

	
	tif* tifPtr = new tif(kjg.rows,kjg.cols,35);
	cv::Mat out = tifPtr->run(kjg, infra);
	cv::imshow("test", out);
	cv::imwrite(name,out);
	cv::waitKey(0);
	return 0;
}