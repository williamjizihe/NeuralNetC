#include "misc.h"
#include <stdlib.h>

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