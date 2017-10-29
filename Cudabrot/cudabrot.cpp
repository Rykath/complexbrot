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
	int widthImg 	= 1000;
	int widthSec	= 10;	// number of sectors on the same image (each sector: widthImg)
	std::string path = "/home/robert/Fractalbrot/test/test.png";
	int iterations 	= 10000;
	float widthC 	= 2.5;
	float cenCr 	= 0.5;
	float cenCi 	= 0.0;
	bool buddha = false;
	if (not buddha){
		if (widthSec <= 1){
			int* data;

			data = sector_mandel(cenCr,cenCi,widthC,widthImg,iterations);

			exportImage(widthImg,widthImg,data,iterations,path);

			free(data);
		}
		else{
			int* data = (int*)malloc(widthSec*widthSec*widthImg*widthImg*sizeof(int));
			int* esc;
			float cr, ci;
			for (int h=0;h<widthSec;h++){
				for (int w=0;w<widthSec;w++){
					cr = cenCr-(w-widthSec/2.0+0.5)*widthC/(float)widthSec;
					ci = cenCi-(h-widthSec/2.0+0.5)*widthC/(float)widthSec;
					esc = sector_mandel(cr,ci,widthC/(float)widthSec,widthImg,iterations);
					for (int H=0;H<widthImg;H++){
						for (int W=0;W<widthImg;W++){
							data[(h*widthImg+H)*widthSec*widthImg+w*widthImg+W] = esc[H*widthImg+W];
						}
					}
					free(esc);
				}
			}
			exportImage(widthImg*widthSec,widthImg*widthSec,data,iterations,path);
			free(data);
		}
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



