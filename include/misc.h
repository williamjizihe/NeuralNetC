#include "ndarray.h"

void data_shuffle(ndarray *data[], int label[], int size);

void read_data(const char* filename, ndarray** images, int* labels, int num, int image_size);

void read_image(const char* filename, ndarray* image, int image_size);