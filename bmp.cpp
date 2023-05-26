#include <fstream>
#include <math.h>


void writeBMP(const char* filename, unsigned char *buffer, int width, int height)
{
	
	unsigned char header[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char headerinfo[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };


	constexpr int pixel_size = 3;
	const int filesize = width * height * pixel_size;

	*(int*)(header + 2) = filesize;

	*(int*)(headerinfo + 4) = width;

	*(int*)(headerinfo + 8) = height;

	FILE* dump = fopen(filename, "wb");

	fwrite(header, 1, 14, dump);
	fwrite(headerinfo, 1, 40, dump);

	fwrite(buffer, 1, width * height * 3, dump);//escupir la imagen de memoria a disco

	fclose(dump);
}