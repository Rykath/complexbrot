OBJS = image_magick.o mandelbrot_simple_cuda.o
LFLAGS = -L/usr/include/ImageMagick -L/usr/include/ImageMagick/Magick++
CFLAGS = -c -I/usr/include/ImageMagick -I/usr/include/ImageMagick/Magick++

mandelbrot: $(OBJS)
	nvcc $(LFLAGS) $(OBJS) -o mandelbrot

mandelbrot_simple_cuda.o: mandelbrot_simple_cuda.cu image_magick.h
	nvcc $(CFLAGS) mandelbrot_simple_cuda.cu

image_magick.o: image_magick.h image_magick.cpp
	g++ $(CFLAGS) image_magick.cpp