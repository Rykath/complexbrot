/*
 *  Created on: Oct 29, 2017
 *      Author: Rykath
 */

#include <string>
#include <math.h>
#include <stdio.h>

__global__ void sequence_mandel(int* escIter, float* cr, float* ci, int maxIter){
	int pos = blockIdx.x*blockDim.x + threadIdx.x;
	
	float zr = 0;
	float zi = 0;
	float zr2 = 0;	// zr squared
	float zi2 = 0;	// zi squared
	
	for (int iter=0; iter<maxIter; iter++){
		zi = 2*zi*zr + ci[pos]; // !! order is important, zi has to be first
		zr = zr2 - zi2 + cr[pos];
		zr2 = zr*zr;
		zi2 = zi*zi;
		if (zr2 + zi2 > 4.0){	// escaping
			escIter[pos] = iter;
			return;
		}
	}
	escIter[pos] = maxIter;
	return;
}

int* sector_mandel(float cenCr, float cenCi, float widthC, int widthSpl, int iterations){
	int* escIter, * escIter_g;	// dual memory: _g is gpu memory
	float* cr,* cr_g;
	float* ci,* ci_g;
	
	int numSpl = widthSpl*widthSpl;
	
	escIter = (int*)malloc(numSpl*sizeof(int));
	cr = (float*)malloc(numSpl*sizeof(float));
	ci = (float*)malloc(numSpl*sizeof(float));
	
	cudaMalloc(&escIter_g, numSpl*sizeof(int));
	cudaMalloc(&cr_g, numSpl*sizeof(float));
	cudaMalloc(&ci_g, numSpl*sizeof(float));

	float resC = widthC/(float)widthSpl;
	for (int h=0; h<widthSpl; h++){
		for (int w=0; w<widthSpl; w++){
			cr[h*widthSpl+w] = (w-widthSpl/2.0)*resC-cenCr;
			ci[h*widthSpl+w] = (h-widthSpl/2.0)*resC-cenCi;
		}
	}
	
	cudaMemcpy(cr_g, cr, numSpl*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ci_g, ci, numSpl*sizeof(float), cudaMemcpyHostToDevice);
    
	sequence_mandel<<<(numSpl+255)/256, 256>>>(escIter_g,cr_g,ci_g,iterations);

	cudaMemcpy(escIter, escIter_g, numSpl*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(escIter_g);
	cudaFree(cr_g);
	cudaFree(ci_g);
	free(cr);
	free(ci);
	
	return escIter;
}

__global__ void sequence_periodicity(int* perIter, float* cr, float* ci, int maxIter, int period, float near){
	int pos = blockIdx.x*blockDim.x + threadIdx.x;
	
	float zr = 0;
	float zi = 0;
	float zr2 = 0;	// zr squared
	float zi2 = 0;	// zi squared
	float pr = 0;	// checking periodicity against
	float pi = 0;
	
	for (int iter=0; iter<maxIter; iter++){
		zi = 2*zi*zr + ci[pos]; // !! order is important, zi has to be first
		zr = zr2 - zi2 + cr[pos];
		zr2 = zr*zr;
		zi2 = zi*zi;
		if (zr2 + zi2 > 4.0){	// escaping
			perIter[pos] = 0;
			return;
		}
		if (iter % period == 0){
			if (abs(pr-zr) < near && abs(pi-zi) < near){
				perIter[pos] = maxIter-iter; // brighter if detected sooner
				return;
			}
			else{
				pr = zr;
				pi = zi;
			}
		}
	}
	perIter[pos] = 0;
	return;
}

int* sector_periodicity(float cenCr, float cenCi, float widthC, int widthSpl, int iterations, int period, float near){
	int* escIter, * escIter_g;	// dual memory: _g is gpu memory
	float* cr,* cr_g;
	float* ci,* ci_g;
	
	int numSpl = widthSpl*widthSpl;
	
	escIter = (int*)malloc(numSpl*sizeof(int));
	cr = (float*)malloc(numSpl*sizeof(float));
	ci = (float*)malloc(numSpl*sizeof(float));
	
	cudaMalloc(&escIter_g, numSpl*sizeof(int));
	cudaMalloc(&cr_g, numSpl*sizeof(float));
	cudaMalloc(&ci_g, numSpl*sizeof(float));

	float resC = widthC/(float)widthSpl;
	for (int h=0; h<widthSpl; h++){
		for (int w=0; w<widthSpl; w++){
			cr[h*widthSpl+w] = (w-widthSpl/2.0)*resC-cenCr;
			ci[h*widthSpl+w] = (h-widthSpl/2.0)*resC-cenCi;
		}
	}
	
	cudaMemcpy(cr_g, cr, numSpl*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ci_g, ci, numSpl*sizeof(float), cudaMemcpyHostToDevice);
    
	sequence_periodicity<<<(numSpl+255)/256, 256>>>(escIter_g,cr_g,ci_g,iterations,period,near);

	cudaMemcpy(escIter, escIter_g, numSpl*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(escIter_g);
	cudaFree(cr_g);
	cudaFree(ci_g);
	free(cr);
	free(ci);
	
	return escIter;
}

__global__ void sequence_buddha(int* escIter, float* escPath, float* cr, float* ci, int maxIter){
	int pos = blockIdx.x*blockDim.x + threadIdx.x;
	
	float zr = 0;
	float zi = 0;
	float zr2 = 0;	// zr squared
	float zi2 = 0;	// zi squared
	
	for (int iter=0; iter<maxIter; iter++){
		zi = 2*zi*zr + ci[pos]; // !! order is important, zi has to be first
		zr = zr2 - zi2 + cr[pos];
		zr2 = zr*zr;
		zi2 = zi*zi;
		escPath[pos*maxIter*2 + iter*2] = zr;
		escPath[pos*maxIter*2 + iter*2 + 1] = zi;
		if (zr2 + zi2 > 4.0){	// escaping
			escIter[pos] = iter;
			return;
		}
	}
	escIter[pos] = maxIter;
	return;
}

void sector_buddha(int** retEsc, float** retPath, float cenCr, float cenCi, float widthC, int widthSpl, int iterations){
	int* escIter, * escIter_g;	// dual memory: _g is gpu memory
	float* cr,* cr_g;
	float* ci,* ci_g;
	float* escPath,* escPath_g;
	
	int numSpl = widthSpl*widthSpl;
	
	escIter = (int*)malloc(numSpl*sizeof(int));
	cr = (float*)malloc(numSpl*sizeof(float));
	ci = (float*)malloc(numSpl*sizeof(float));
	escPath = (float*)malloc(numSpl*sizeof(int)*2*iterations);
	
	cudaMalloc(&escIter_g, numSpl*sizeof(int));
	cudaMalloc(&cr_g, numSpl*sizeof(float));
	cudaMalloc(&ci_g, numSpl*sizeof(float));
	cudaMalloc(&escPath_g, numSpl*sizeof(float)*2*iterations);

	float resC = widthC/(float)widthSpl;
	for (int h=0; h<widthSpl; h++){
		for (int w=0; w<widthSpl; w++){
			cr[h*widthSpl+w] = (w-widthSpl/2.0)*resC-cenCr;
			ci[h*widthSpl+w] = (h-widthSpl/2.0)*resC-cenCi;
		}
	}
	
	cudaMemcpy(cr_g, cr, numSpl*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ci_g, ci, numSpl*sizeof(float), cudaMemcpyHostToDevice);
    
	sequence_buddha<<<(numSpl+255)/256, 256>>>(escIter_g,escPath_g,cr_g,ci_g,iterations);

	cudaMemcpy(escIter, escIter_g, numSpl*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(escPath, escPath_g, numSpl*sizeof(float)*2*iterations, cudaMemcpyDeviceToHost);
	
	cudaFree(escIter_g);
	cudaFree(cr_g);
	cudaFree(ci_g);
	cudaFree(escPath_g);
	
	free(cr);
	free(ci);
	
	*retEsc = escIter;
	*retPath = escPath;
	return;
}
