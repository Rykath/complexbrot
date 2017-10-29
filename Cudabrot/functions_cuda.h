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
// (return escapes, return paths, ...)
void sector_buddha(int**,float**,float,float,float,int,int);

#endif /* FUNCTIONS_CUDA_H_ */
