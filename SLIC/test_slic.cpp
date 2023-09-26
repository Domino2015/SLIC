#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
using namespace std;
using namespace cv;

#include "slic.h"

int main(int argc, char* argv[]) {
    /* Load the image and convert to Lab colour space. */
    Mat image = imread("./dog.png", IMREAD_COLOR);
    if (image.empty()) {
        printf("Could not open image file.\n");
        return -1;
    }
    Mat lab_image = image.clone();
    cvtColor(image, lab_image, COLOR_BGR2Lab);

    /* Yield the number of superpixels and weight-factors from the user. */
    int w = image.cols, h = image.rows;
    int nr_superpixels = 200;
    int nc = 40;

    double step = sqrt((w * h) / (double)nr_superpixels);

    /* Perform the SLIC superpixel algorithm. */
    Slic slic;
    slic.generate_superpixels(lab_image, step, nc);
    slic.create_connectivity(lab_image);

    /* Display the contours and show the result. */
    slic.display_contours(image, Scalar(255, 0, 0));
    //imshow("result", image);
    //waitKey(0);
    imwrite("./dog_segmentation.png", image);

    return 0;
}
