/*
 *  Created on: Oct 22, 2017
 *      Author: Rykath
 */

#include <string>
#include <stdio.h>

#include "image_magick.h"
#include "mandelbrot_simple_cuda.h"

int main(){
	int width = 1000;
	int height = 1000;
	std::string path = "/home/robert/Fractalbrot/test/test.png";
	int iterations = 100000;
	float widthC = 2.5;

	int* data;

	data = mandelbrot(width,height,iterations,widthC,path);

	exportImage(width,height,data,iterations,path);

	free(data);

	printf("done\n");
	return 0;
}



