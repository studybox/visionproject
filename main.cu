/*--------------------------------------------------------------------------
 
 File Name:         main.cpp
 Date Created:      2016/06/30
 Date Modified:     2016/08/18
 
 Author:            Eric Cristofalo
 Contact:           eric.cristofalo@gmail.com
 
 Description:       Depth From Focus Algorithm
 
 -------------------------------------------------------------------------*/

// Include Libraries
#include <iostream>
#include <stdio.h>
#include <fstream>
//#include <boost/filesystem.hpp>
#include <iomanip>
//#include <unistd.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "DfF.h"
#include "filepathconfig.h"


// Namespaces
using namespace cv;
using namespace std;

//Try this
const double CLK_TCK = 1000.0;

int main(int argc, const char * argv[]) {
    
    // File Path Initialization
    string folderPath = INPUT_PATH;
    string outputFolderPath = OUTPUT_PATH;
    cout << folderPath<< endl;
    string imageName;
    int numImageDigits = 4;
    int startInd = 50;
    int endInd = 85;
    int totalInd = endInd-startInd+1;
    
    // Figure Initialization
    namedWindow("Current Image", CV_WINDOW_AUTOSIZE);
    namedWindow("Image Gradient", CV_WINDOW_AUTOSIZE);
    namedWindow("Depth Image", CV_WINDOW_AUTOSIZE);
    
    // Variable Initialization
    int loopInd = 0;
    Mat im, imGray, im1, im2, imAvg;
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;
    cv::Feature2D * detector;
    int numFeatures = 5000;
    detector = new SiftFeatureDetector(numFeatures);
    vector<Mat> H_i0;
    Mat Gx, Gy, absGx, absGy, Gxy;
    vector<Mat> imTotal, GxyTotal;
    Mat  imIndex, imMaxVal, imDepth, imDepthDense;
    
    // Timer
    clock_t start00, end00;
    start00 = clock();
    clock_t start01,end01;
    
    // Compute Image Alignement and Focal Measure for Entire Image Stack
    for (int imInd=startInd; imInd<=endInd; imInd++) {

        // Read Current Image
        ostringstream ss;
        ss << setw(numImageDigits) << setfill('0') << imInd;
        string curNum = ss.str();
        
        imageName = folderPath+curNum+".jpg";
        //cout<< imageName <<endl;
        im = imread(imageName);
        //if (im.data==NULL) cout<<"empty"<<endl;
        //imshow( "Current Image", im );  
        
        cout << "Image Index: " << imInd << endl;
       
//        // Resize Image
//        float resizeScale = 0.6;
//        Size dSize = Size(round(im.cols*resizeScale),round(im.rows*resizeScale));
//        resize(im, im, dSize);
       
        // Initialize First Image Variables
        if (imInd==startInd) {
            // Current Image
            im1 = im;
            // Initialize Depth Matrices
            imIndex = (-1)*Mat::ones(im.size(), CV_32F);
            imDepth = Mat::zeros(im.size(), CV_32F);
            imMaxVal = Mat::zeros(im.size(), CV_32F);
            imDepthDense = Mat::zeros(im.size(), CV_32F);
            // Extract Features
            detector->detect(im1, keypoints1);
            detector->compute(im1, keypoints1, descriptor1);
            // Homography
            H_i0.push_back(Mat::eye(3,3,CV_64F));
            // Average Image for Superpixel
            im.copyTo(imAvg);
        }
   
        
        else {
            // Loop Index
            loopInd++;
            // Pushback Current Image to Previous
            im2.release(); keypoints2.empty(); descriptor2.release();
            im2 = im1;
            
            keypoints2.swap(keypoints1);
            
            descriptor2 = descriptor1;
            
            // Extract Features
            
            im1.release(); keypoints1.empty(); descriptor1.release();
            im1 = im;
            detector->detect(im1, keypoints1);
            detector->compute(im1, keypoints1, descriptor1);
            // Compute Interimage Homography
            
            Mat H_12 = Mat::eye(3,3,CV_64F);
            vector<DMatch> goodMatches;
            
            computeHomography(H_12, goodMatches, keypoints1, keypoints2, descriptor1, descriptor2);
            // Total Homography
            
            H_i0.push_back(H_12*H_i0[loopInd-1]);
            // Perspective Transform
            Mat imWarped;
            warpPerspective(im1, imWarped, H_i0[loopInd], im1.size()); // (object, scene, homography, destination size)
            imshow("Current Image",imWarped);
            // Average Image for Superpixel
            imAvg += im;
            
        }
        
    
        
        // Compute Focus Measure on Current Image
        int ksize = 1;
        int scale = 1;
        int delta = 1;
        int ddepth = CV_16S;
        cvtColor(im, imGray, CV_BGR2GRAY);
        // Gradient X
        Sobel(imGray, Gx, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
//        convertScaleAbs(Gx, absGx);
        Gx.convertTo(absGx, CV_32F);
        // Gradient Y
        Sobel(imGray, Gy, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
//        convertScaleAbs(Gy, absGy);
        Gy.convertTo(absGy, CV_32F);
        // Total Gradient (approximate)
        Gxy.release();
        Mat Gxy = Mat::zeros(im.size(),CV_32F);
        Gxy = 0.5*absGx+0.5*absGy;
//        addWeighted(absGx, 0.5, absGy, 0.5, 0, Gxy);
        
        // Total Gradient Image
        warpPerspective(Gxy, Gxy, H_i0[loopInd], im.size()); // (object, scene, homography, destination size)
        GxyTotal.push_back(Gxy);
        
        // Start Timing the Focus Metric
        start01 = clock();

        // Compute Sharpest Pixel Per 3D Window
        int windowSize = 1; // must be odd valued
        int edgeVal = (windowSize-1)/2;
        float focalThresh = 10.0;
        if (imInd>startInd+1 && imInd<endInd) {
            // Compute the Average 3D Gradient Image
            Mat GAvg = Mat::zeros(windowSize,windowSize,CV_32F);
            GAvg = GxyTotal[loopInd-2] + GxyTotal[loopInd-1] + GxyTotal[loopInd];
            GAvg = GAvg/3.0;
            for (int i=edgeVal; i<(im.rows-edgeVal); i++) {
                for (int j=edgeVal; j<(im.cols-edgeVal); j++) {
//                    // Compute the Average Window
//                    Mat curWindow = Mat::zeros(windowSize,windowSize,CV_32F);
//                    Rect roi(j-edgeVal, i-edgeVal, windowSize, windowSize);
//                    for (int k=(loopInd-2); k<=loopInd; k++) {
//                        Mat Gxy_cur = GxyTotal[k];
//                        curWindow = curWindow+Gxy_cur(roi);
//                    }
//                    curWindow = curWindow/3.0;
//                    Scalar avgTemp = mean(curWindow);
//                    float curValue = float(avgTemp[0]);
                    float curValue = GAvg.at<float>(i,j);
                    // Determine Index of Maximizing Value
                    float prevValue = imMaxVal.at<float>(i,j);
                    if (curValue>prevValue && curValue>focalThresh) {
                        imMaxVal.at<float>(i,j) = curValue;
                        imIndex.at<float>(i,j) = float(loopInd-1);
                    }
                }
            }
            
        }
        
        // End Timer
        end01 = clock();
        cout<<"Focus Measure Computation Time (ms): "<<(static_cast<double>(end01)-start01)/CLK_TCK<<endl;
        
        // Display Results
        Mat GxyDisp, imIndexDisp, imMaxValDisp;
        double max, min;
        minMaxIdx(Gxy, &min, &max);
        //    convertScaleAbs(Gxy, GxyDisp, 255/max);
        scale2RGB(Gxy, GxyDisp);
        
        imshow("Image Gradient", GxyDisp);
        
        
        minMaxIdx(imIndex, &min, &max);
        //    convertScaleAbs(imIndex, imIndexDisp, 255/max);
        scale2RGB(imIndex, imIndexDisp);
        imshow("Index Image", imIndexDisp);
        minMaxIdx(imMaxVal, &min, &max);
        //    convertScaleAbs(imMaxVal, imMaxValDisp, 255/max);
        scale2RGB(imMaxVal, imMaxValDisp);
        imshow("Max Val Image", imMaxValDisp);
        //if (Gxy.data == NULL) cout<<"gxempty"<<endl;
          
//        // Saving Images
//        imageName = folderPath+"/Gradient_Images"+"/image_"+curNum+".jpg";
//        imwrite(imageName, imGrad);
//        imageName = folderPath+"/Depth_Images"+"/image_"+curNum+".jpg";
//        imwrite(imageName, imDepth);
        
        // Display Images
        if (cvWaitKey(1)==27) {return 0;}

        //if (Gxy.data != NULL) cout<<"gxnotempty2"<<endl;
    }
    //if (Gxy.data == NULL) cout<<"gxempty3"<<endl;
        
    // Filtered Indexed Depth Map Via Blur Filter
//    blur(imIndex, imIndex, Size(3,3), Point(-1,-1));
    
    // Filtered Indexed Depth Map with imMaxVal
    
    // Display Results
    Mat GxyDisp, imIndexDisp, imMaxValDisp;
    double max, min;
    minMaxIdx(Gxy, &min, &max);
//    convertScaleAbs(Gxy, GxyDisp, 255/max);
    
    scale2RGB(Gxy, GxyDisp);
    
    //imshow("Image Gradient", GxyDisp);
    minMaxIdx(imIndex, &min, &max);
//    convertScaleAbs(imIndex, imIndexDisp, 255/max);
    scale2RGB(imIndex, imIndexDisp);
    if (imIndex.data == NULL) cout<<"inempty"<<endl;
    imshow("Index Image", imIndexDisp);
    
    minMaxIdx(imMaxVal, &min, &max);
//    convertScaleAbs(imMaxVal, imMaxValDisp, 255/max);
    scale2RGB(imMaxVal, imMaxValDisp);
    if (imMaxVal.data == NULL) cout<<"maempty"<<endl;
    imshow("Max Val Image", imMaxValDisp);
    
    // My Superpixel Method
    // TO DO STILL
    
    // End Timer
    end00 = clock();
    cout<<"Total Algorithm Time (ms): "<<(static_cast<double>(end00)-start00)/CLK_TCK<<endl;
    
    // Hold at End of Program
    if (cvWaitKey(0)==27) {return 0;}
    
    return 0;
}

