/*
 * mandelbrot_simple.cpp
 *
 *  Created on: Oct 20, 2017
 *      Author: Rykath
 */

#include <string>

#include "image_magick.h"

// CUDA kernel
__global__
void mandelsequence(int* escIter,double* w, double* h){
	int I = 10000;
	double Cscale = 0.0025;		// 2/800  --  complex scale per calc unit
	double Cr = -0.5 + *w * Cscale;
	double Ci = *h * Cscale;

	double Zr;
	double Zi;
	double Zr2;
	double Zi2;

	double escThresh = 4.0;

	Zr = 0;
	Zi = 0;
	Zr2 = Zr*Zr;
	Zi2 = Zi*Zi;
	if (Zr2 + Zi2 > escThresh){
		// escape = true;
		*escIter = 0;
		return;
	}
	for (int i=1;i<I;i++){
		Zi = 2*Zi*Zr + Ci; // !! order is important, Zi has to be first
		Zr = Zr2 - Zi2 + Cr;
		Zr2 = Zr*Zr;
		Zi2 = Zi*Zi;
		if (Zr2 + Zi2 > escThresh){
			// escape = true;
			*escIter = i;
			return;
		}
	}
	// escape = false;
	*escIter = I;
	return;
}

int run(){
	int width = 800;
	int height = 600;
	std::string path = "/home/robert/Fractalbrot/test/test.png";


	//int data [width*height];
	int* data;
	double* W,* H;
	int* iter;
	cudaMalloc(&data, width*height*sizeof(int));
	cudaMalloc(&W, width*height*sizeof(float));
	cudaMalloc(&H, width*height*sizeof(float));
	cudaMalloc(&iter, width*height*sizeof(int));

	for (int h=0; h<height; h++){
		for (int w=0; w<width; w++){
			W[h*width+w] = w-width/2;
			H[h*width+w] = h-height/2;
		}
	}
	mandelsequence<<<1, 1>>>(iter,W,H);

	cudaDeviceSynchronize();
	
	exportImage(width,height,data,10000,path);

	cudaFree(data);
	cudaFree(W);
	cudaFree(H);
	cudaFree(iter);

	printf("done\n");
	return 0;
}

int main(){
	return run();
}
