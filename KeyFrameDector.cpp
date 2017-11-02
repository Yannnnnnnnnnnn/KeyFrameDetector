#include <iostream>
#include <string>
#include <vector>
#include <sstream>
using namespace std;
      
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
using namespace cv;   

void getMeanandStd(const vector<double> &num,double &mean,double &stdenv)
{
    mean = 0;
    stdenv = 0;
    //mean and std
    double sum = std::accumulate(std::begin(num),std::end(num),0.0);
    mean = sum/num.size();
    double accum = 0.0;
    std::for_each (std::begin(num),std::end(num),[&](const double d)
    {
        accum += (d-mean)*(d-mean);
    });
    stdenv = sqrt(accum/(num.size()-1));
}

string int2str(const int int_temp)
{
    stringstream stream;
    stream<<int_temp;
    return stream.str();
}

double calcClarity(const Mat &image)
{
    Mat dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    Laplacian(image,dst,ddepth,kernel_size,scale,delta,BORDER_DEFAULT);
    Scalar mean,stddev;
    meanStdDev(dst,mean,stddev);
    return stddev.val[0];

}

int main( int argc, char** argv )
{
    VideoCapture cap;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);
    const int MAX_COUNT = 500;
    
    std::string video_file_path = "1.mp4";
    std::string output_file_path = "/home/yan/Desktop/KeyFrameDetector/build/output/";

    cap.open(video_file_path);
    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }
    
    //int frame_width = cv::GetCaptureProperty(cap,CV_CAP_PROP_FRAME_WIDTH);
    //int frame_height = cv::GetCaptureProperty(cap,CV_CAP_PROP_FRAME_HEIGHT);

    Mat image,current_frame;
    Mat key_frame;
    int key_frame_count = 0;
    Mat current_frame_gray,key_frame_gray;
    vector<Point2f> key_frame_points,current_frame_points;
    vector<Point2f> cornor_points;

    int a= 0;
    while(1)
    {
        cout<<a++<<endl;

        cap >> current_frame;
        if( current_frame.empty() )
            break;

        
            
        //init
        //look as the first frame as the first key frame
        if(key_frame.empty())
        {
            current_frame.copyTo(key_frame);
            cvtColor(key_frame, key_frame_gray, COLOR_BGR2GRAY);

            cornor_points.clear();
            goodFeaturesToTrack(key_frame_gray, cornor_points, MAX_COUNT, 0.01, 10);
            cornerSubPix(key_frame_gray, cornor_points, subPixWinSize, Size(-1,-1), termcrit);

            key_frame_points.clear();
            key_frame_points.assign(cornor_points.begin(),cornor_points.end());

        }
        else
        {
            cvtColor(current_frame, current_frame_gray, COLOR_BGR2GRAY);
            
            cornor_points.clear();
            goodFeaturesToTrack(current_frame_gray, cornor_points, MAX_COUNT, 0.01, 10);
            cornerSubPix(current_frame_gray, cornor_points, subPixWinSize, Size(-1,-1), termcrit);

            current_frame_points.clear();
            current_frame_points.assign(cornor_points.begin(),cornor_points.end());

            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(current_frame_gray, key_frame_gray,current_frame_points, key_frame_points, status, err, winSize,3, termcrit, 0, 0.001);
            
            //calc the distance
            std::vector<double> moving_distance;
            for(int i = 0; i < key_frame_points.size(); i++ )
            {
                if( status[i] )
                {
                    moving_distance.push_back(sqrt( pow(key_frame_points[i].x-current_frame_points[i].x,2)+pow(key_frame_points[i].y-current_frame_points[i].y,2) ));
                }
            }
            
            double mean_all=0,stdenv_all=0;
            getMeanandStd(moving_distance,mean_all,stdenv_all);

            //only keep the point in 2&stdenv
            std::vector<double> moving_distance_filter;
            for(int i=0;i<moving_distance.size();i++)
            {
                if( abs(moving_distance[i]-mean_all)<2*stdenv_all )
                {
                    moving_distance_filter.push_back(moving_distance[i]);
                } 
            }

            //recompute the mean and std
            double mean_filter=0,stdenv_filter=0;
            getMeanandStd(moving_distance_filter,mean_filter,stdenv_filter);

            std::cout<<"Moving: "<<mean_all<<"\t"<<mean_filter<<std::endl;


            //dealing
            if(mean_filter<50)
            {
                /*
                //compare blurry and select the clear one
                double key_frame_clarity_index = calcClarity(key_frame_gray);
                double current_frame_clarity_index = calcClarity(current_frame_gray);
                cout<<"clarity:"<<key_frame_clarity_index<<"\t"<<current_frame_clarity_index<<endl;
                if(current_frame_clarity_index>key_frame_clarity_index)
                {
                    cv::swap(key_frame, current_frame);
                    cvtColor(key_frame, key_frame_gray, COLOR_BGR2GRAY);
    
                    cornor_points.clear();
                    goodFeaturesToTrack(key_frame_gray, cornor_points, MAX_COUNT, 0.01, 10);
                    cornerSubPix(key_frame_gray, cornor_points, subPixWinSize, Size(-1,-1), termcrit);
    
                    key_frame_points.clear();
                    key_frame_points.assign(cornor_points.begin(),cornor_points.end());
                }
                */
            }
            else
            {
                //save the keyframe and set the current frame as the next key frame
                string image_name = int2str(key_frame_count++)+".jpg";
                cv::imwrite(output_file_path+image_name,key_frame);

                cv::swap(key_frame, current_frame);
                cvtColor(key_frame, key_frame_gray, COLOR_BGR2GRAY);

                cornor_points.clear();
                goodFeaturesToTrack(key_frame_gray, cornor_points, MAX_COUNT, 0.01, 10);
                cornerSubPix(key_frame_gray, cornor_points, subPixWinSize, Size(-1,-1), termcrit);

                key_frame_points.clear();
                key_frame_points.assign(cornor_points.begin(),cornor_points.end());
            }

        }
    }
    return 0;
}
