#include <iostream>
#include <string>
#include <vector>
using namespace std;
      
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
using namespace cv;   



static void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
            "Using OpenCV version " << CV_VERSION << endl;
    cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
    cout << "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tr - auto-initialize tracking\n"
            "\tc - delete all the points\n"
            "\tn - switch the \"night\" mode on/off\n"
            "To add/remove a feature point click it\n" << endl;
}

int main( int argc, char** argv )
{
    VideoCapture cap;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);
    const int MAX_COUNT = 500;
    bool needToInit = false;
    help();
    cv::CommandLineParser parser(argc, argv, "{@input|0|}");
    string input = parser.get<string>("@input");
    if( input.size() == 1 && isdigit(input[0]) )
        cap.open(input[0] - '0');
    else
        cap.open(input);
    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }
    namedWindow( "LK Demo", 1 );
    Mat gray, prevGray, image, frame;
    Mat current_gray,next_gray;
    vector<Point2f> points[2];
    vector<Point2f> cornor_points;
    while(1)
    {
        cap >> frame;
        if( frame.empty() )
            break;
        frame.copyTo(image);
        cvtColor(image, next_gray, COLOR_BGR2GRAY);
        
        cornor_points.clear();
        goodFeaturesToTrack(next_gray, cornor_points, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
        cornerSubPix(next_gray, cornor_points, subPixWinSize, Size(-1,-1), termcrit);
        points[0].assign(cornor_points.begin(),cornor_points.end());

        if(!current_gray.empty())
        {
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(current_gray, next_gray, points[1], points[0], status, err, winSize,3, termcrit, 0, 0.001);
            for(int i = 0; i < points[1].size(); i++ )
            {
                if( !status[i] )
                    continue;
                cv::line(image,points[1][i],points[0][i], Scalar(0,255,0),3);
            }
        }
        cv::swap(next_gray, current_gray);
        
        points[1].assign(cornor_points.begin(),cornor_points.end());
        

        imshow("LK Demo", image);
        char c = (char)waitKey(10);
        if( c == 27 )
            break;

    }
    return 0;
}
