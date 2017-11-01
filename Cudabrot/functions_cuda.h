/*
 *  Created on: Oct 29, 2017
 *      Author: Rykath
 */

#ifndef FUNCTIONS_CUDA_H_
#define FUNCTIONS_CUDA_H_

// sectors:
// squares
// (center Cr, center Ci, width Complex, width Samples, maximum iterations)
int* sector_mandel(float,float,float,int,int);
// (..., period to check for, exactness
int* sector_periodicity(float,float,float,int,int,int,float);
// (return escapes, return paths, ...)
void sector_buddha(int**,float**,float,float,float,int,int);

#endif /* FUNCTIONS_CUDA_H_ */
