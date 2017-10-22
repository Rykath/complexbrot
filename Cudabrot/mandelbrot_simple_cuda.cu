/*
 *  Created on: Oct 20, 2017
 *      Author: Rykath
 */

#include <string>
#include <math.h>
#include <stdio.h>

// CUDA kernel
__global__
void mandelsequence(int* escIter,int* w, int* h, int I, float Cscale){
	int pos = blockIdx.x*blockDim.x + threadIdx.x;
	
	float Cr = -0.5 + w[pos] * Cscale;
	float Ci = h[pos] * Cscale;

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
		escIter[pos] = 0;
		return;
	}
	for (int i=1;i<I;i++){
		Zi = 2*Zi*Zr + Ci; // !! order is important, Zi has to be first
		Zr = Zr2 - Zi2 + Cr;
		Zr2 = Zr*Zr;
		Zi2 = Zi*Zi;
		if (Zr2 + Zi2 > escThresh){
			// escape = true;
			escIter[pos] = i;
			return;
		}
	}
	// escape = false;
	escIter[pos] = I;
	return;
}

int* mandelbrot(int width, int height, int iterations, float widthC, std::string path){
	float Cscale = widthC/(float) width;

	int* data, * data_g;
	int* W,* W_g;
	int* H,* H_g;
	
	data = (int*)malloc(width*height*sizeof(int));
	W = (int*)malloc(width*height*sizeof(int));
	H = (int*)malloc(width*height*sizeof(int));
	
	cudaMalloc(&data_g, width*height*sizeof(int));
	cudaMalloc(&W_g, width*height*sizeof(int));
	cudaMalloc(&H_g, width*height*sizeof(int));

	for (int h=0; h<height; h++){
		for (int w=0; w<width; w++){
			W[h*width+w] = w-width/2;
			H[h*width+w] = h-height/2;
			data[h*width+w] = 0;
		}
	}
	
	cudaMemcpy(W_g, W, width*height*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(H_g, H, width*height*sizeof(int), cudaMemcpyHostToDevice);
    
	mandelsequence<<<(width*height+255)/256, 256>>>(data_g,W_g,H_g,iterations,Cscale);

	cudaMemcpy(data, data_g, width*height*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(data_g);
	cudaFree(W_g);
	cudaFree(H_g);
	free(W);
	free(H);

	printf("calc done\n");
	return data;
}
