//#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
int main() {
    // Open the video file
    cv::VideoCapture cap("Misaka Mikoto Railgun.mp4");

    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the video file." << std::endl;
        return -1;
    }

    // Enable OpenCL in OpenCV 4.x
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU)) {
        std::cerr << "Error: Unable to create OpenCL context." << std::endl;
        return -1;
    }

    cv::ocl::setUseOpenCL(true);

    while (true) {
        // Read the current frame
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        // Convert the frame to grayscale
        cv::UMat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Apply Canny edge detection using OpenCL
        cv::UMat edges;
        cv::Canny(gray, edges, 50, 150);

        // Download the result from the GPU
        cv::Mat edges_cpu;
        edges.copyTo(edges_cpu);

        // Display the result
        cv::imshow("Video Processing with OpenCL", edges_cpu);

        // Break the loop if 'q' key is pressed
        if (cv::waitKey(30) == 'q')
            break;
    }

    // Release the VideoCapture and close all windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
