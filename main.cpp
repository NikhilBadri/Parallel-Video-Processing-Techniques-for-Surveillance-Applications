//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//C
#include <stdio.h>
#include <string.h>
//C++
#include <iostream>
#include <sstream>

//cuda
#include <cuda_runtime.h>
//extern
#include "VideoFilters.h"

using namespace cv;
using namespace std;

//PARAMETERS
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

//global variables
Mat frame; //current frame
Mat MOG; // foreground mask MOG method
Mat MOG2; // foreground amsk MOG1 method
Mat GMG; //foregroung GMG
Mat GMG2;
Mat mout;
Mat frame_median;
Mat Nframe;

//Ptr<BackgroundSubtractorMOG> pMOG; //MOG Background subtractor
Ptr<BackgroundSubtractorMOG2> pMOG2; //MOG2 Background subtarctor
//Ptr<BackgroundSubtractorGMG> pGMG; //GMG background subtractor
Ptr<BackgroundSubtractorGMG> pGMG2;

int key;
int i = 0;

//Calculate blocks
int iDivUp(int n, int blockDim){
	int nBlocks = n / blockDim;
	if (n % blockDim)
		return nBlocks++;

	return nBlocks;
}

extern "C" uchar4 CUDA_MotionVec(float* new_image_dev, float* old_image_dev,uchar4 *newf, int imageW, int imageH, dim3 grid, dim3 threads);
//extern "C" void CUDA_MeanFilter(uchar4 *Image_dev, uchar4 *pframe, int imageW, int imageH, dim3 grid, dim3 threads);
void videoprocess(char* videoFilename);
Mat meanfilter(Mat);
Mat motionvector(Mat,Mat,Mat);

void help()
{
	cout << "*************************************************************************************" << endl;
	cout << "aguments can be given as:" << endl << ".\<exe> <video filelocation>" << endl;
	cout << "*************************************************************************************" << endl;
}



int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cerr << "incorrect input" << endl;
		return EXIT_FAILURE;
	}
	// creating windows
	namedWindow("Frame", CV_WINDOW_AUTOSIZE);	
	namedWindow("MOG2", CV_WINDOW_AUTOSIZE);	
	namedWindow("GMG", CV_WINDOW_AUTOSIZE);
	namedWindow("mean filter", CV_WINDOW_AUTOSIZE);

	//background subtractor 	
	pMOG2 = new BackgroundSubtractorMOG2();
	pGMG2 = new BackgroundSubtractorGMG();

	videoprocess(argv[1]);

	//destroy windows
	destroyAllWindows();
	return EXIT_SUCCESS;
}

void videoprocess(char* videoFilename)
{
	//video capture
	VideoCapture cap(videoFilename);
	if (!cap.isOpened())
	{
		cerr << "Unable to open the file" << videoFilename << endl; //file open error
		exit(EXIT_FAILURE);
	}
	
	/* Gather some video statistics */
	double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	double TotalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);

	while ((char)key != 'q' && (char)key != 27)
	{
		
		if (!cap.read(frame))
		{
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			exit(EXIT_FAILURE);
		}
			
		//background updation		
		pMOG2->operator()(frame, MOG2);		
		pGMG2->operator()(frame, GMG2);
		imshow("Frame", frame);
		
		Nframe = meanfilter(frame);
		
		if (i == 0)
		{
				frame_median = Nframe;
				i++;
		}
		
		mout = motionvector(frame_median, Nframe, Nframe);
		frame_median = mout;
		
		//show the current frame and the fg masks
		imshow("MOG2", MOG2);
		imshow("GMG", GMG2);

		key = waitKey(30);
	}

	cap.release(); //delete capture object
}

Mat meanfilter(Mat frame)
{
	///***************************************************
	//*    Mean Filter
	//***************************************************/

	/*Size of Image*/
	int imageW = frame.cols;
	int imageH = frame.rows;

	size_t size = imageW*imageH*sizeof(uchar4);

	/* Create a 4 channel data structures */
	cv::Mat frame_Original4c_median;
	frame_Original4c_median.create(imageH, imageW, CV_8UC(4));
	cv::cvtColor(frame, frame_Original4c_median, CV_BGR2BGRA, 0);	

	/* Create device Variables & device memory */
	uchar4 *Image_dev;	
	cudaMalloc((void **)&Image_dev, size);	
	CUDA_CreateMemoryArray(imageW, imageH);

	/*Copy Memory (Host-->Device)*/
	cudaMemcpy(Image_dev, frame_Original4c_median.data, size, cudaMemcpyHostToDevice);
	
	/*Define the size of the grid and thread blocks*/
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, 1);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y), 1);

	CUDA_MemcpyToArray(Image_dev, imageW, imageH);
	CUDA_BindTextureToArray();

	/* Mean Filter Launch the Kernel Function*/
	CUDA_MeanFilter(Image_dev, imageW, imageH, grid, threads);

	/*Copy Memory (Device-->Host)*/
	cudaMemcpy(frame_Original4c_median.data, Image_dev, size, cudaMemcpyDeviceToHost);
	
	/*Device*/
	cudaFree(Image_dev);	
	CUDA_FreeArrays();

	/*  Conditional checks for data and bounds of video*/
	if (frame_Original4c_median.data != NULL)
	{
		cv::imshow("mean filter", frame_Original4c_median);
	}

	return frame_Original4c_median;
}


Mat motionvector(Mat oldframe, Mat newframe, Mat f)
{
	/*Size of Image*/
	int imageW = f.cols;
	int imageH = f.rows;

	size_t size = imageW*imageH*sizeof(uchar4);
	size_t fsize = imageW*imageH*sizeof(float);

	/* Create device Variables & device memory */
	uchar4 *newf;
	float *new_image_data, *old_image_data;

	cudaMalloc((void **)&new_image_data, fsize);
	cudaMalloc((void **)&old_image_data, fsize);
	cudaMalloc((void **)&newf, size);
	
	/*Copy Memory (Host-->Device)*/
	cudaMemcpy(newf, f.data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(old_image_data, oldframe.data, fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(new_image_data, newframe.data, fsize, cudaMemcpyHostToDevice);
	
	/*Define the size of the grid and thread blocks*/
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, 1);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y), 1);
		
	/* Mean Filter Launch the Kernel Function*/
	CUDA_MotionVec(new_image_data, old_image_data, newf, imageW, imageH, grid, threads);
		
	/*Copy Memory (Device-->Host)*/
	cudaMemcpy(oldframe.data, old_image_data, fsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(newframe.data, new_image_data, fsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(f.data, newf, size, cudaMemcpyDeviceToHost);
	
	/* Clean Device
	*
	* INSERT CODE HERE
	*/	
	cudaFree(old_image_data);
	cudaFree(new_image_data);
	cudaFree(newf);
	
	if (f.data != NULL)
	{
		cv::imshow("motion frame", f);
	}
	
	return(newframe);
}
