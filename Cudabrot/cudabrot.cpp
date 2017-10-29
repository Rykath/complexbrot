/*
 *  Created on: Oct 22, 2017
 *      Author: Rykath
 */

#include <string>
#include <stdio.h>
#include <math.h>

#include "image_magick.h"
#include "functions_cuda.h"

int main(){
	int widthImg 	= 100;
	std::string path = "/home/robert/Fractalbrot/test/test.png";
	int iterations 	= 10;
	float widthC 	= 2.5;
	float cenCr 	= 0.5;
	float cenCi 	= 0.0;
	bool buddha = true;

	if (not buddha){
		int* data;

		data = sector_mandel(cenCr,cenCi,widthC,widthImg,iterations);

		exportImage(widthImg,widthImg,data,iterations,path);

		free(data);
	}
	else {
		int* esc;
		float* paths;
		int data [widthImg*widthImg];
		for (int x=0; x<widthImg*widthImg; x++){
			data[x] = 0;
		}
		float wcr, hci;
		float resSpl = widthImg/widthC;

		sector_buddha(&esc,&paths,cenCr,cenCi,widthC,widthImg,iterations);

		for (int h=0; h<widthImg; h++){
			for (int w=0; w<widthImg; w++){
				if (esc[h*widthImg+w] < iterations){
					for (int i=0; i<esc[h*widthImg+w]; i++){
						wcr = paths[2*((h*widthImg+w)*iterations+i)]+cenCr;
						hci = paths[2*((h*widthImg+w)*iterations+i)+1]+cenCi;
						wcr = wcr*resSpl;
						hci = hci*resSpl;
						data[(int)round(hci)*widthImg+(int)wcr] += 1;
					}
				}
			}
		}
		int maxValue = 0;
		for (int x=0; x<widthImg*widthImg; x++){
			if (data[x] > maxValue){
				maxValue = data[x];
			}
		}
		exportImage(widthImg,widthImg,data,maxValue,path);

		free(paths);
		free(esc);
	}

	printf("done\n");
	return 0;
}



