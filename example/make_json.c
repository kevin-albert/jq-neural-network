// 
// Reads the MNIST database and outputs JSON suitable for example.jq
// usage: ./make_json -t [train-size] [-n test-size]
//  where `train-size` is between 1 and 60000 and `test-size` is between 1 and
//  10000 
//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define LBL_HEADER_SIZE     (8)
#define IMG_HEADER_SIZE    (16)
#define LBL_SIZE            (1)
#define IMG_SIZE        (28*28)

void process_set(const char *label_file, const char *image_file, 
                 const int num_records, const int is_train);
FILE *open_or_fail(const char *filename);

int main(int argc, char **argv) {
    int c;
    int t = 60000;
    int n = 10000;
    while ((c = getopt(argc, argv, "t:n:h")) != -1) {
        switch (c) {
            case 't':
                t = atoi(optarg);
                if (t < 1 || t > 60000) {
                    fprintf(stderr, "invalid training set size " 
                                    "(must be between 1 and 60000): %s\n", optarg);
                    exit(EXIT_FAILURE);
                }
                break;
            case 'n':
                n = atoi(optarg);
                if (n < 1 || n > 10000) {
                    fprintf(stderr, "invalid data set size " 
                                    "(must be between 1 and 10000): %s\n", optarg);
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                fprintf(c == 'h' ? stdout : stderr,
                        "usage: %s [-t train-size] [-n test-size]\n", argv[0]);
                exit(c == 'h' ? 0 : EXIT_FAILURE);
        }
    }

    puts("\"hello\""); // first input record is ignored
    process_set("train-labels-idx1-ubyte", "train-images-idx3-ubyte", t, 1);
    process_set("t10k-labels-idx1-ubyte",  "t10k-images-idx3-ubyte",  n, 0);
}


FILE *open_or_fail(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        char err[100];
        sprintf(err, "unable to open %s", filename);
        perror(err);
        exit(EXIT_FAILURE);
    }
    return file;
}


void process_set(const char *label_file, const char *image_file, 
                 const int num_records, const int is_train) {
    FILE *labels = open_or_fail(label_file);
    FILE *images = open_or_fail(image_file);

    if (fseek(labels, LBL_HEADER_SIZE, SEEK_SET)) {
        perror("fseek() failed");
        exit(EXIT_FAILURE);
    }

    if (fseek(images, IMG_HEADER_SIZE, SEEK_SET)) {
        perror("fseek() failed");
        exit(EXIT_FAILURE);
    }

    unsigned char label;
    unsigned char image[IMG_SIZE];
    for (int i = 0; i < num_records; ++i) {
        
        // read a label
        if (fread(&label, LBL_SIZE, 1, labels) != 1) {
            perror("unable to read label");
            exit(EXIT_FAILURE);
        }
        // read an image
        if (fread(image,  IMG_SIZE, 1, images) != 1) {
            perror("unable to read image");
            exit(EXIT_FAILURE);
        }

        printf("{\"train\":");
        if (is_train) printf("true");
        else printf("false");

        printf(",\"expected\":[");
        for (int j = 0; j < 10; ++j) {
            if (j) putc(',', stdout);
            if (label == j) putc('1', stdout);
            else putc('0', stdout);
        }
        printf("],\"input\":[");

        for (int j = 0; j < IMG_SIZE; ++j) {
            if (j) putc(',', stdout);
            printf("%f", ((float) image[j])/256);
        }

        printf("]}\n");
    }

}