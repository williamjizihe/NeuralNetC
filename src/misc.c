#include "misc.h"
#include <stdlib.h>
#include <stdio.h>

void data_shuffle(ndarray *data[], int label[], int size){
    if (size > 1) {
        int i;
        for (i = 0; i < size - 1; i++) {
            int j = i + rand() / (RAND_MAX / (size - i) + 1);
            ndarray* t = data[j];
            data[j] = data[i];
            data[i] = t;

            int l = label[j];
            label[j] = label[i];
            label[i] = l;
        }
    }
}

void read_data(const char* filename, ndarray** images, int* labels, int num, int image_size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Cannot open file %s\n", filename);
        exit(1);
    }

    // Read each line from the file
    for (int i = 0; i < num; i++) {
        char image_file[255];
        int label;
        fscanf(file, "%s %d", image_file, &label);

        labels[i] = label;
        images[i] = nda_zero(2, (int[]){image_size * image_size, 1});

        // Open the image file and read the data
        FILE* image_file_handle = fopen(image_file, "r");
        if (image_file_handle == NULL) {
            printf("Cannot open file %s\n", image_file);
            exit(1);
        }
        
        for (int j = 0; j < image_size * image_size; j++) {
            int pixel_value;
            fscanf(image_file_handle, "%d", &pixel_value);
            images[i]->data[j] = (float)pixel_value;
        }
        fclose(image_file_handle);
    }

    fclose(file);
}

void read_image(const char* filename, ndarray* image, int image_size){
    FILE* image_file_handle = fopen(filename, "r");
    if (image_file_handle == NULL) {
        printf("Cannot open file %s\n", filename);
        exit(1);
    }
    
    for (int j = 0; j < image_size * image_size; j++) {
        int pixel_value;
        fscanf(image_file_handle, "%d", &pixel_value);
        image->data[j] = (float)pixel_value;
    }
    fclose(image_file_handle);
}