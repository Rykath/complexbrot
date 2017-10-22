/*
 *  Created on: Oct 20, 2017
 *      Author: Rykath
 */

#ifndef IMAGE_MAGICK_H_
#define IMAGE_MAGICK_H_

#include <string>
#include <Magick++.h>
using namespace Magick;

void exportImage(int,int,int*,int,std::string);

#endif /* IMAGE_MAGICK_H_ */
