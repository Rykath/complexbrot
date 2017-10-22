/*
 *  Created on: Oct 20, 2017
 *      Author: Rykath
 */

#include <string>
#include <stdio.h>

#include "image_magick.h"

int mandelsequence(double Cr, double Ci, double Kr, double Ki, double I){
	double Zr;
	double Zi;
	double Zr2;
	double Zi2;

	double escThresh = 4.0;

	Zr = Kr;
	Zi = Ki;
	Zr2 = Zr*Zr;
	Zi2 = Zi*Zi;
	if (Zr2 + Zi2 > escThresh){
		// escape = true;
		return 0;
	}
	for (int i=1;i<I;i++){
		Zi = 2*Zi*Zr + Ci; // !! order is important, Zi has to be first
		Zr = Zr2 - Zi2 + Cr;
		Zr2 = Zr*Zr;
		Zi2 = Zi*Zi;
		if (Zr2 + Zi2 > escThresh){
			// escape = true;
			return i;
		}
	}
	// escape = false;
	return I;
}

int main(){
	int width = 800;
	int height = 600;
	std::string path = "/home/robert/Fractalbrot/test/test.png";
	int iterations = 1000;

	double Cr_c = -0.5;
	double Cr_x = 1;
	double Ci_c = 0;
	double Ci_y = 1;
	double Cscale = 2.0/800.0;		// complex scale per calc unit

	int data [width*height];
	for (int h=0; h<height; h++){
		for (int w=0; w<width; w++){
			data[h*width+w] = mandelsequence(Cr_c + Cr_x*(w-width/2)*Cscale,Ci_c + Ci_y*(h-height/2)*Cscale,0.0,0.0,iterations);
		}
	}

	exportImage(width,height,data,iterations,path);

	printf("done\n");
	return 0;
}
