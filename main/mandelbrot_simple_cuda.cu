/*
 * mandelbrot_simple.cpp
 *
 *  Created on: Oct 20, 2017
 *      Author: Rykath
 */

#include <string>
#include <math.h>
#include <stdio.h>

#include "image_magick.h"

// CUDA kernel
__global__
void mandelsequence(int* escIter,float* w, float* h, int I, float Cscale){
	float Cr = -0.5 + *w * Cscale;
	float Ci = *h * Cscale;

	float Zr;
	float Zi;
	float Zr2;
	float Zi2;

	float escThresh = 4.0;

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

//	void exportImage(int width, int height, int* data, int iterations, std::string path){
//		FILE* fptr;
//		char* file = new char[128];
//		sprintf(file,"%s",path.c_str());
//		fptr = fopen(file,"wb");
//		char col;
//		for (int i=0; i<height*width; i++){
//			col = char(round(data[i]/(float)iterations * 255));
//			fwrite(&col,1,1,fptr);
//		}
//		fclose(fptr);
//	}

int main(){
	int width = 8;
	int height = 6;
	std::string path = "/home/robert/Fractalbrot/test/test.png";
	float Cscale = 0.0025;	// 2:800  -- scale for complex plane
	int iterations = 1000;

	int* data, * data_g;
	float* W,* W_g;
	float* H,* H_g;
	
	data = (int*)malloc(width*height*sizeof(int));
	W = (float*)malloc(width*height*sizeof(float));
	H = (float*)malloc(width*height*sizeof(float));
	
	cudaMalloc(&data_g, width*height*sizeof(int));
	cudaMalloc(&W_g, width*height*sizeof(float));
	cudaMalloc(&H_g, width*height*sizeof(float));

	for (int h=0; h<height; h++){
		for (int w=0; w<width; w++){
			W[h*width+w] = w-width/2;
			H[h*width+w] = h-height/2;
		}
	}
	
	cudaMemcpy(W_g, W, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(H_g, H, width*height*sizeof(float), cudaMemcpyHostToDevice);
    
	mandelsequence<<<1, 1>>>(data,W,H,iterations,Cscale);

	cudaMemcpy(data, data_g, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	
	exportImage(width,height,data,iterations,path);

	cudaFree(data_g);
	cudaFree(W_g);
	cudaFree(H_g);
	free(data);
	free(W);
	free(H);

	printf("done\n");
	return 0;
}
