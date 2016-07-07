#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
#include <omp.h>

using namespace std;

typedef char my_char;
typedef int my_int;
typedef float my_float;
typedef bool my_bool;
typedef long long my_clock;
typedef double my_double;
#define vec2dd vector< vector<my_double> > 
#define vec2di vector< vector<my_int> > 
#define vec2db vector< vector<my_bool> > 
#define NC 40
#define NR 10

vec2di regions;
vec2dd distances;
vec2dd centers;
vector<my_int> center_counts;
my_int step, ns;
//g++ -ggdb `pkg-config --cflags opencv` -o main main.cpp `pkg-config --libs opencv` /home/franci/Escritorio/UCSP Master/Parallel/SuperPixel/SLIC-Superpixels-master/im1.png

my_clock wall_clock_time() {
#ifdef __linux__
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (my_clock)(tp.tv_nsec + (my_clock)tp.tv_sec * 1000000000ll);
#else
	#warning "Your timer resoultion might be too low. Compile on Linux and link with librt"
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (my_clock)(tv.tv_usec * 1000 + (my_clock)tv.tv_sec * 1000000000ll);
#endif
}


void generate_baatz(IplImage *image, my_int s);
void create_connectivity(IplImage *image);
void display_center_grid(IplImage *image, CvScalar colour);
void display_contours(IplImage *image, CvScalar colour);
void draw_region_colour(IplImage *image);
double compute_dist(int ci, CvPoint pixel, CvScalar colour);
CvPoint find_local_minimum(IplImage *image, CvPoint center);
void clear_data();
void init_image(IplImage *image);






int main() {
    my_clock start, end;
	start = wall_clock_time();//start measure time
	
    IplImage *image = cvLoadImage("dog.png", 1);
    IplImage *lab_image = cvCloneImage(image);
    cvCvtColor(image, lab_image, CV_BGR2Lab);
    //cvSaveImage("lab.png", image);
    my_int w = image->width, h = image->height;
    my_double step = sqrt((w * h) / (my_double) 30000);
    
    generate_baatz(lab_image, step);
    
    create_connectivity(lab_image);
    draw_region_colour(image);
    display_contours(image, CV_RGB(255,255,255));
    cvSaveImage("dogx.png", image);
    
    end = wall_clock_time();//end measure time
	fprintf(stderr, "Image Segmentation took %1.2f seconds\n", ((my_float)(end - start))/1000000000);
    
}

void generate_baatz(IplImage *image, my_int s) 
{
    step = s;
    ns = s;
    clear_data();
    init_image(image);
    
    for (my_int i = 0; i < NR; i++) {
        for (my_int j = 0; j < image->width; j++) {
            for (my_int k = 0;k < image->height; k++) {
                distances[j][k] = FLT_MAX;
            }
        }
        omp_set_num_threads(4);
	#pragma omp parallel for
        for (my_int j = 0; j < (my_int) centers.size(); j++) {
            for (my_int k = centers[j][3] - step; k < centers[j][3] + step; k++) {
                for (my_int l = centers[j][4] - step; l < centers[j][4] + step; l++) {
                
                    if (k >= 0 && k < image->width && l >= 0 && l < image->height) {
                        CvScalar colour = cvGet2D(image, l, k);
                        my_double d = compute_dist(j, cvPoint(k,l), colour);
                        if (d < distances[k][l]) {
                            distances[k][l] = d;
                            regions[k][l] = j;
                        }
                    }
                }
            }
        }

        for (my_int j = 0; j < (my_int) centers.size(); j++) {
            centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
            center_counts[j] = 0;
        }
        for (my_int j = 0; j < image->width; j++) {
            for (my_int k = 0; k < image->height; k++) {
                my_int c_id = regions[j][k];
                
                if (c_id != -1) {
                    CvScalar colour = cvGet2D(image, k, j);
                    
                    centers[c_id][0] += colour.val[0];
                    centers[c_id][1] += colour.val[1];
                    centers[c_id][2] += colour.val[2];
                    centers[c_id][3] += j;
                    centers[c_id][4] += k;
                    
                    center_counts[c_id] += 1;
                }
            }
        }
        omp_set_num_threads(4);
	#pragma omp parallel for
        for (my_int j = 0; j < (my_int) centers.size(); j++) {
            centers[j][0] /= center_counts[j];
            centers[j][1] /= center_counts[j];
            centers[j][2] /= center_counts[j];
            centers[j][3] /= center_counts[j];
            centers[j][4] /= center_counts[j];
        }
    }
}

void clear_data() {
    regions.clear();
    distances.clear();
    centers.clear();
    center_counts.clear();
}


void init_image(IplImage *image) 
{
    for (my_int i = 0; i < image->width; i++) { 
        vector<my_int> cr;
        vector<my_double> dr;
        for (my_int j = 0; j < image->height; j++) {
            cr.push_back(-1);
            dr.push_back(FLT_MAX);
        }
        regions.push_back(cr);
        distances.push_back(dr);
    }

    for (my_int i = step; i < image->width - step/2; i += step) {
        for (my_int j = step; j < image->height - step/2; j += step) {
            vector<my_double> center;
            CvPoint nc = find_local_minimum(image, cvPoint(i,j));
            CvScalar colour = cvGet2D(image, nc.y, nc.x);
            center.push_back(colour.val[0]);
            center.push_back(colour.val[1]);
            center.push_back(colour.val[2]);
            center.push_back(nc.x);
            center.push_back(nc.y);
            centers.push_back(center);
            center_counts.push_back(0);
        }
    }

}


CvPoint find_local_minimum(IplImage *image, CvPoint center) {
    my_double min_grad = FLT_MAX;
    CvPoint loc_min = cvPoint(center.x, center.y);
    
    for (my_int i = center.x-1; i < center.x+2; i++) {
        for (my_int j = center.y-1; j < center.y+2; j++) {
            CvScalar c1 = cvGet2D(image, j+1, i);
            CvScalar c2 = cvGet2D(image, j, i+1);
            CvScalar c3 = cvGet2D(image, j, i);

            my_double i1 = c1.val[0];
            my_double i2 = c2.val[0];
            my_double i3 = c3.val[0];

            if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3,2)) < min_grad) {
                min_grad = fabs(i1 - i3) + fabs(i2 - i3);
                loc_min.x = i;
                loc_min.y = j;
            }
        }
    }
    
    return loc_min;
}


double compute_dist(my_int ci, CvPoint pixel, CvScalar colour) {
    my_double dc = sqrt(pow(centers[ci][0] - colour.val[0], 2) + pow(centers[ci][1]
            - colour.val[1], 2) + pow(centers[ci][2] - colour.val[2], 2));
    my_double ds = sqrt(pow(centers[ci][3] - pixel.x, 2) + pow(centers[ci][4] - pixel.y, 2));
    
    return sqrt(pow(dc / NC, 2) + pow(ds / ns, 2));

}


void create_connectivity(IplImage *image) {
    my_int label = 0, adjlabel = 0;
    const my_int lims = (image->width * image->height) / ((int)centers.size());
    
    const my_int dx4[4] = {-1,  0,  1,  0};
	const my_int dy4[4] = { 0, -1,  0,  1};

    vec2di new_regions;
    for (my_int i = 0; i < image->width; i++) { 
        vector<my_int> nc;
        for (my_int j = 0; j < image->height; j++) {
            nc.push_back(-1);
        }
        new_regions.push_back(nc);
    }

    for (my_int i = 0; i < image->width; i++) {
        for (my_int j = 0; j < image->height; j++) {
            if (new_regions[i][j] == -1) {
                vector<CvPoint> elements;
                elements.push_back(cvPoint(i, j));

                for (my_int k = 0; k < 4; k++) {
                    my_int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
                    
                    if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
                        if (new_regions[x][y] >= 0) {
                            adjlabel = new_regions[x][y];
                        }
                    }
                }
                
                my_int count = 1;
                for (my_int c = 0; c < count; c++) {
                    for (my_int k = 0; k < 4; k++) {
                        my_int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
                        
                        if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
                            if (new_regions[x][y] == -1 && regions[i][j] == regions[x][y]) {
                                elements.push_back(cvPoint(x, y));
                                new_regions[x][y] = label;
                                count += 1;
                            }
                        }
                    }
                }

                if (count <= lims >> 2) {
                    for (my_int c = 0; c < count; c++) {
                        new_regions[elements[c].x][elements[c].y] = adjlabel;
                    }
                    label -= 1;
                }
                label += 1;
            }
        }
    }
}


void display_center_grid(IplImage *image, CvScalar colour) {
    for (my_int i = 0; i < (my_int) centers.size(); i++) {
        cvCircle(image, cvPoint(centers[i][3], centers[i][4]), 2, colour, 2);
    }
}

void display_contours(IplImage *image, CvScalar colour) {
    const my_int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const my_int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	vector<CvPoint> contours;
	vec2db istaken;
	for (my_int i = 0; i < image->width; i++) { 
        vector<my_bool> nb;
        for (my_int j = 0; j < image->height; j++) {
            nb.push_back(false);
        }
        istaken.push_back(nb);
    }

    for (my_int i = 0; i < image->width; i++) {
        for (my_int j = 0; j < image->height; j++) {
            int nr_p = 0;
            for (my_int k = 0; k < 8; k++) {
                my_int x = i + dx8[k], y = j + dy8[k];
                
                if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
                    if (istaken[x][y] == false && regions[i][j] != regions[x][y]) {
                        nr_p += 1;
                    }
                }
            }

            if (nr_p >= 2) {
                contours.push_back(cvPoint(i,j));
                istaken[i][j] = true;
            }
        }
    }

    for (my_int i = 0; i < (my_int)contours.size(); i++) {
        cvSet2D(image, contours[i].y, contours[i].x, colour);
    }
}


void draw_region_colour(IplImage *image)
{
    vector<CvScalar> colours(centers.size());
    
    for (my_int i = 0; i < image->width; i++) {
        for (my_int j = 0; j < image->height; j++) {
            my_int index = regions[i][j];
            CvScalar colour = cvGet2D(image, j, i);
            
            colours[index].val[0] += colour.val[0];
            colours[index].val[1] += colour.val[1];
            colours[index].val[2] += colour.val[2];
        }
    }

    for (my_int i = 0; i < (my_int)colours.size(); i++) {
        colours[i].val[0] /= center_counts[i];
        colours[i].val[1] /= center_counts[i];
        colours[i].val[2] /= center_counts[i];
    }

    for (my_int i = 0; i < image->width; i++) {
        for (my_int j = 0; j < image->height; j++) {
            CvScalar ncolour = colours[regions[i][j]];
            cvSet2D(image, j, i, ncolour);
        }
    }
}
