#include <fstream>


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

	int i, j = 0;
	while (j < height)
	{
		i = 0;
		while (i < width)
		{
			fwrite(&buffer[(i + j * height) * 3], 1, 3, dump);
			i++;
		}
		j++;
	}

	fclose(dump);
}