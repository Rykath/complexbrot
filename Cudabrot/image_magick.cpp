/*
 * image_magick.cpp
 *
 *  Created on: Oct 20, 2017
 *      Author: Rykath
 */

#include <math.h>
#include <string>
#include <Magick++.h>
using namespace Magick;

#include "image_magick.h"

void exportImage(int width, int height, int* data, int iterations, std::string path){
	// data is int-array
	// -- size: height*width
	// -- format: height*[data from one row (width)]

	Image image (Geometry(width,height),Color("white"));	// Magick: image-object
	Pixels pixels (image);									// Magick: pixels-object from image
	PixelPacket* pixp = pixels.get(0,0,width,height);		// Magick: pointer to pixel-array

	for (int h=0; h<height; h++){
		for (int w=0; w<width; w++){
			int gray = round(log((float)data[h*width+w])/log(iterations) * MaxRGB);
			Color color = Color(gray,gray,gray,0);

			*(pixp+h*width+w) = color;
		}
	}
	pixels.sync();		// Magick: update pixels

	image.magick("png");
	image.write(path);
}
