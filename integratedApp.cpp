#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <vector>
#include <deque>
#include <atomic>
#include <chrono>
#include <sched.h>
#include <unistd.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

using namespace cv;
using namespace std;
using namespace chrono;

// Line structure
struct Line
{
    float slope;
    float intercept;
    Line(float s, float i) : slope(s), intercept(i) {}
};

struct ThreadArgs
{
    Mat frame;
    Mat hsvFrame;
    Mat grayFrame;
    CascadeClassifier *stopSignCascade;
    CascadeClassifier *trafficLightCascade;
    CascadeClassifier *pedestrianCascade;
    vector<Rect> stop_signs;
    vector<Vec4i> filteredLines;
    vector<Rect> pedestrians;

    vector<Rect> traffic_lights;

    int feature;
    int core_id;
};

// Function prototypes
void detectStopSigns(Mat &frame, CascadeClassifier &stopSignCascade, const Mat &hsvFrame);
void detectTrafficLights(Mat &frame, CascadeClassifier &trafficLightCascade, const Mat &hsvFrame);
void detectPedestrians(Mat &frame, CascadeClassifier &pedestrianCascade);
void detectLanes(Mat &frame, const Mat &grayFrame);
void processFrame();

vector<Vec4i> detectLaneLines(const Mat &edges);
vector<Vec4i> filterLines(const vector<Vec4i> &lines, const Size &dimensions);
void drawLines(Mat &image, const vector<Vec4i> &lines, Scalar color = Scalar(0, 255, 0));
void visualizeLanes(Mat &image, const vector<Vec4i> &lines, const Size &dimensions);
void checkLaneDeparture(Mat &image, const vector<Vec4i> &lines, const Size &dimensions);
Line averageLines(const vector<Line> &candidates);

Mat adjustGamma(const Mat &img, double gamma);

std::deque<Line> left_lines;
std::deque<Line> right_lines;
std::deque<bool> left_detected_history;
std::deque<bool> right_detected_history;
const int history_length = 20;
const int missing_threshold = 17;

std::queue<Mat> frameQueue;
std::mutex queueMutex;
std::condition_variable queueCondVarsend;
std::condition_variable queueCondVarreceive;
int capacity_ = 300;

std::mutex laneMutex;
std::condition_variable laneCondVar;

std::mutex pedMutex;
std::condition_variable pedCondVar;

std::mutex stopMutex;
std::condition_variable stopCondVar;

std::mutex trafMutex;
std::condition_variable trafCondVar;

std::mutex frameprocessedMutex;
std::vector<bool> processedFrames = {false, false, false, false};

std::atomic<bool> stopProcessing(false);

Mat redMask;

ThreadArgs threadArgs;
std::mutex threadArgsMutex;
CascadeClassifier stopSignCascade, trafficLightCascade, pedestrianCascade;

Scalar lower_red1(0, 100, 100), upper_red1(25, 255, 255);
Scalar lower_red2(160, 100, 100), upper_red2(180, 255, 255);

Scalar lower_red(0, 150, 150), upper_red(30, 255, 255);
Scalar lower_green(50, 150, 150), upper_green(100, 255, 255);

void processFrame()
{
    Mat clonedFrame_, hsvFrame_, grayFrame_;
    int feature;
    {
        std::unique_lock<std::mutex> lock(threadArgsMutex);
        clonedFrame_ = threadArgs.frame.clone();
        hsvFrame_ = threadArgs.hsvFrame.clone();
        grayFrame_ = threadArgs.grayFrame.clone();
        feature = threadArgs.feature;
    }

    switch (feature)
    {
    case 0:
        detectStopSigns(clonedFrame_, stopSignCascade, hsvFrame_);
        {
            std::unique_lock<std::mutex> lock(stopMutex);
            stopCondVar.notify_one();
        }

        {
            std::unique_lock<std::mutex> lock(frameprocessedMutex);
            processedFrames[0] = true;
        }
        break;
    case 1:
        detectTrafficLights(clonedFrame_, trafficLightCascade, hsvFrame_);
        {
            std::unique_lock<std::mutex> lock(trafMutex);
            trafCondVar.notify_one();
        }

        {
            std::unique_lock<std::mutex> lock(frameprocessedMutex);
            processedFrames[1] = true;
        }

        break;
    case 2:
        detectPedestrians(clonedFrame_, pedestrianCascade);
        {
            std::unique_lock<std::mutex> lock(pedMutex);
            pedCondVar.notify_one();
        }

        {
            std::unique_lock<std::mutex> lock(frameprocessedMutex);
            processedFrames[2] = true;
        }

        break;
    case 3:
        detectLanes(clonedFrame_, grayFrame_);
        {
            std::unique_lock<std::mutex> lock(laneMutex);
            laneCondVar.notify_one();
        }

        {
            std::unique_lock<std::mutex> lock(frameprocessedMutex);
            processedFrames[3] = true;
        }

        break;
    }
}

inline void set_core_affinity(pthread_t tid, int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    int rc = pthread_setaffinity_np(tid,
                                    sizeof(cpu_set_t), &cpuset);

    if (rc != 0)
    {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
}
std::atomic<bool> closed_(false);

void threadFunction(int feature, int core_id, CascadeClassifier &stopSignCascade, CascadeClassifier &trafficLightCascade, CascadeClassifier &pedestrianCascade)
{
    while (!stopProcessing)
    {
        Mat frame;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (frameQueue.empty() && !closed_)
            {
                queueCondVarreceive.wait(lock);
            }
            if (frameQueue.empty() && closed_)
            {
                throw Exception();
            }

            frame = frameQueue.front();
        }

        Mat hsvFrame, grayFrame;
        cvtColor(frame, hsvFrame, COLOR_BGR2HSV);
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        {
            std::lock_guard<std::mutex> lock(threadArgsMutex);
            threadArgs.frame = frame;
            threadArgs.hsvFrame = hsvFrame;
            threadArgs.grayFrame = grayFrame;
            threadArgs.feature = feature;
        }

        processFrame();
    }
}

void frameLoaderThread(VideoCapture cap)
{
    while (!stopProcessing)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        {
            std::unique_lock<std::mutex> lock(queueMutex);

            if (frameQueue.size() >= capacity_)
            {
                queueCondVarsend.wait(lock);
            }

            frameQueue.push(std::move(frame));
            queueCondVarreceive.notify_one();
        }
    }
}

void detectStopSigns(Mat &frame, CascadeClassifier &stopSignCascade, const Mat &hsvFrame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    Mat red_mask1, red_mask2;

    inRange(hsvFrame, lower_red1, upper_red1, red_mask1);
    //inRange(hsvFrame, lower_red2, upper_red2, red_mask2);
    redMask = red_mask1;// | red_mask2;

    Rect roi(0, 0, frame_gray.cols, frame_gray.rows / 2);
    Mat frame_roi = frame_gray(roi);

    vector<Rect> stop_signs;
    stopSignCascade.detectMultiScale(
        frame_roi,
        stop_signs,
        1.4,
        5,
        0 | CASCADE_SCALE_IMAGE,
        Size(50, 50));

    {
        std::lock_guard<std::mutex> lock(threadArgsMutex);
        threadArgs.stop_signs = stop_signs;
    }
}

void detectTrafficLights(Mat &frame, CascadeClassifier &trafficLightCascade, const Mat &hsvFrame)
{
    Mat frame_gray, adjustedROIFrame;
    double gamma = 0.2;
    adjustedROIFrame = adjustGamma(frame, gamma);

    cvtColor(adjustedROIFrame, frame_gray, COLOR_BGR2GRAY);

    vector<Rect> traffic_lights;
    trafficLightCascade.detectMultiScale(frame_gray, traffic_lights, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(5, 7));

    // Define the ROI for traffic lights (top half of the frame)
    Rect roi(0, 0, frame.cols, frame.rows / 2);
    Mat roiFrame = frame_gray(roi);

    {
        std::lock_guard<std::mutex> lock(threadArgsMutex);
        threadArgs.traffic_lights = traffic_lights;
    }
}

void detectPedestrians(Mat &frame, CascadeClassifier &pedestrianCascade)
{
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    vector<Rect> pedestrians;

    pedestrianCascade.detectMultiScale(gray, pedestrians, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 60));

    {
        std::lock_guard<std::mutex> lock(threadArgsMutex);
        threadArgs.pedestrians = pedestrians;
    }
}

void detectLanes(Mat &frame, const Mat &grayFrame)
{
    Mat edges, maskedGray, regionOfInterest = Mat::zeros(frame.size(), CV_8U);
    Mat blur;
    GaussianBlur(grayFrame, blur, Size(5, 5), 0);

    rectangle(regionOfInterest, Point(0, frame.rows / 2), Point(frame.cols, frame.rows), Scalar(255), FILLED);
    blur.copyTo(maskedGray, regionOfInterest);

    Canny(maskedGray, edges, 50, 150);

    vector<Vec4i> lines = detectLaneLines(edges);
    auto filteredLines = filterLines(lines, frame.size());

    {
        std::lock_guard<std::mutex> lock(threadArgsMutex);
        threadArgs.filteredLines = filteredLines;
    }
}

vector<Vec4i> detectLaneLines(const Mat &edges)
{
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 10, 250);
    return lines;
}

vector<Vec4i> filterLines(const vector<Vec4i> &lines, const Size &dimensions)
{
    vector<Line> left_candidates, right_candidates;
    for (const auto &line : lines)
    {
        Point pt1(line[0], line[1]), pt2(line[2], line[3]);
        float slope = (float)(pt2.y - pt1.y) / (pt2.x - pt1.x);
        float intercept = pt1.y - slope * pt1.x;
        if (abs(slope) > 0.5 && abs(slope) < 2)
        {
            if (slope < 0 && pt1.x < dimensions.width / 2)
            {
                left_candidates.push_back(Line(slope, intercept));
            }
            else if (slope > 0 && pt2.x > dimensions.width / 2)
            {
                right_candidates.push_back(Line(slope, intercept));
            }
        }
    }

    if (!left_candidates.empty())
    {
        if (left_lines.size() >= 5)
            left_lines.pop_front();
        left_lines.push_back(averageLines(left_candidates));
        if (left_detected_history.size() >= history_length)
            left_detected_history.pop_front();
        left_detected_history.push_back(true);
    }
    else
    {
        if (left_detected_history.size() >= history_length)
            left_detected_history.pop_front();
        left_detected_history.push_back(false);
    }

    if (!right_candidates.empty())
    {
        if (right_lines.size() >= 5)
            right_lines.pop_front();
        right_lines.push_back(averageLines(right_candidates));
        if (right_detected_history.size() >= history_length)
            right_detected_history.pop_front();
        right_detected_history.push_back(true);
    }
    else
    {
        if (right_detected_history.size() >= history_length)
            right_detected_history.pop_front();
        right_detected_history.push_back(false);
    }

    Line left_avg = averageLines(vector<Line>(left_lines.begin(), left_lines.end()));
    Line right_avg = averageLines(vector<Line>(right_lines.begin(), right_lines.end()));

    vector<Vec4i> averagedLines;
    if (!left_lines.empty())
    {
        int y1 = dimensions.height;
        int y2 = dimensions.height * 2 / 3;
        int x1 = (y1 - left_avg.intercept) / left_avg.slope;
        int x2 = (y2 - left_avg.intercept) / left_avg.slope;
        averagedLines.push_back(Vec4i(x1, y1, x2, y2));
    }
    if (!right_lines.empty())
    {
        int y1 = dimensions.height;
        int y2 = dimensions.height * 2 / 3;
        int x1 = (y1 - right_avg.intercept) / right_avg.slope;
        int x2 = (y2 - right_avg.intercept) / right_avg.slope;
        averagedLines.push_back(Vec4i(x1, y1, x2, y2));
    }

    return averagedLines;
}

void drawLines(Mat &image, const vector<Vec4i> &lines, Scalar color)
{
    for (const auto &line : lines)
    {
        cv::line(image, Point(line[0], line[1]), Point(line[2], line[3]), color, 10, LINE_AA);
    }
}

void visualizeLanes(Mat &image, const vector<Vec4i> &lines, const Size &dimensions)
{
    if (lines.size() >= 2)
    {
        Mat overlay = image.clone();
        vector<Point> poly_points;

        poly_points.push_back(Point(lines[0][0], lines[0][1]));
        poly_points.push_back(Point(lines[0][2], lines[0][3]));
        poly_points.push_back(Point(lines[1][2], lines[1][3]));
        poly_points.push_back(Point(lines[1][0], lines[1][1]));

        fillConvexPoly(overlay, poly_points, Scalar(0, 255, 0, 100));

        double alpha = 0.3;
        addWeighted(overlay, alpha, image, 1 - alpha, 0, image);
    }
}

void checkLaneDeparture(Mat &image, const vector<Vec4i> &lines, const Size &dimensions)
{
    Point midBottom(dimensions.width / 2, dimensions.height);
    Point midTop(dimensions.width / 2, dimensions.height * 2 / 3);
    line(image, midBottom, midTop, Scalar(255, 0, 0), 5, LINE_AA);

    bool left_detected = false, right_detected = false;
    Point leftBottom, rightBottom;
    if (lines.size() >= 2)
    {
        leftBottom = Point(lines[0][0], lines[0][1]);
        rightBottom = Point(lines[1][0], lines[1][1]);
        left_detected = true;
        right_detected = true;
    }
    else if (lines.size() == 1)
    {
        if (lines[0][0] < dimensions.width / 2)
        {
            leftBottom = Point(lines[0][0], lines[0][1]);
            left_detected = true;
        }
        else
        {
            rightBottom = Point(lines[0][0], lines[0][1]);
            right_detected = true;
        }
    }

    const int distance_threshold = dimensions.width / 5;
    if (left_detected)
    {
        int leftDistance = abs(leftBottom.x - midBottom.x);
        if (leftDistance < distance_threshold)
        {
            putText(image, "Left Lane Departure Warning!", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
    }
    if (right_detected)
    {
        int rightDistance = abs(rightBottom.x - midBottom.x);
        if (rightDistance < distance_threshold)
        {
            putText(image, "Right Lane Departure Warning!", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
    }

    int left_missing_count = count(left_detected_history.begin(), left_detected_history.end(), false);
    int right_missing_count = count(right_detected_history.begin(), right_detected_history.end(), false);

    if (left_missing_count > missing_threshold || right_missing_count > missing_threshold)
    {
        putText(image, "Lane Departure/Missing Warning!", Point(50, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    }
}

Line averageLines(const vector<Line> &candidates)
{
    float sum_slope = 0, sum_intercept = 0;
    int count = candidates.size();
    if (count > 0)
    {
        for (const Line &line : candidates)
        {
            sum_slope += line.slope;
            sum_intercept += line.intercept;
        }
        return Line(sum_slope / count, sum_intercept / count);
    }
    return Line(0, 0);
}

Mat adjustGamma(const Mat &img, double gamma)
{
    Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
    {
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    Mat res = img.clone();
    LUT(img, lookUpTable, res);

    return res;
}
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Usage: ./integratedApp <input_video>" << endl;
        return -1;
    }

    string inputVideoPath = argv[1];

    if (!stopSignCascade.load("stop_sign.xml"))
    {
        cerr << "Error loading stop sign cascade file." << endl;
        return -1;
    }
    if (!trafficLightCascade.load("stop_light.xml"))
    {
        cerr << "Error loading traffic light cascade file." << endl;
        return -1;
    }
    if (!pedestrianCascade.load("ped.xml"))
    {
        cerr << "Error loading pedestrian cascade file." << endl;
        return -1;
    }

    VideoCapture cap(inputVideoPath);
    if (!cap.isOpened())
    {
        cout << "Error: Could not open video." << endl;
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    int frameCount = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    std::thread frameLoader(frameLoaderThread, std::move(cap));

    VideoWriter outputVideo("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frameWidth, frameHeight));

    int core_ids[4] = {2, 3, 4, 5};
    std::thread threads[4];

    {
        std::lock_guard<std::mutex> lock(threadArgsMutex);
        for (int i = 0; i < 4; ++i)
        {
            threads[i] = std::thread(threadFunction, i, core_ids[i], std::ref(stopSignCascade), std::ref(trafficLightCascade), std::ref(pedestrianCascade));
            set_core_affinity(threads[i].native_handle(), core_ids[i]);
        }
    }

    auto start_time = steady_clock::now();
    int frame_count = 0;

    while (frame_count < frameCount)
    {
        Mat combinedFrame;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCondVarreceive.wait(lock, []
                                     { return !frameQueue.empty(); });

            combinedFrame = frameQueue.front();
        }

        {
            std::unique_lock<std::mutex> lock(laneMutex);
            laneCondVar.wait(lock, []
                             { return true; });
        }

        {
            std::unique_lock<std::mutex> lock(stopMutex);

            stopCondVar.wait(lock, []
                             { return true; });
        }

        {
            std::unique_lock<std::mutex> lock(trafMutex);

            trafCondVar.wait(lock, []
                             { return true; });
        }

        {
            std::unique_lock<std::mutex> lock(pedMutex);

            pedCondVar.wait(lock, []
                            { return true; });
        }

        {
            std::unique_lock<std::mutex> lock1(frameprocessedMutex);

            processedFrames[0] = false;
            processedFrames[1] = false;
            processedFrames[2] = false;
            processedFrames[3] = false;
        }

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            frameQueue.pop();
            queueCondVarsend.notify_one();
        }

        auto stop_signs = threadArgs.stop_signs;
        auto filteredLines = threadArgs.filteredLines;
        auto pedestrians = threadArgs.pedestrians;
        auto traffic_lights = threadArgs.traffic_lights;

        for (Rect &sign : stop_signs)
        {
            Mat sign_region = redMask(sign);
            double red_area = countNonZero(sign_region);
            double red_area_percentage = (red_area / (sign.width * sign.height)) * 100;
            if (red_area_percentage > 0.0002)
            {
                rectangle(combinedFrame, sign, Scalar(0, 0, 255), 2); // Yellow color
                putText(combinedFrame, "Stop Sign", Point(sign.x, sign.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
            }
        }

        for (const auto &loc : pedestrians)
        {
            if (loc.width > 80 || loc.height > 160)
                continue; // Filter out large boxes
            rectangle(combinedFrame, loc, Scalar(0, 255, 0), 2);
            putText(combinedFrame, "Pedestrian", Point(loc.x, loc.y - 10), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 255, 0), 1);
        }

        drawLines(combinedFrame, filteredLines);
        visualizeLanes(combinedFrame, filteredLines, combinedFrame.size());
        checkLaneDeparture(combinedFrame, filteredLines, combinedFrame.size());

        for (const Rect &tl : traffic_lights)
        {
            if (tl.width > 50 || tl.height > 50)
                continue; // Filter out large boxes

            Mat tl_roi = combinedFrame(tl);
            Mat hsv_roi;
            cvtColor(tl_roi, hsv_roi, COLOR_BGR2HSV);

            Mat red_tl_mask1, red_tl_mask2, green_tl_mask;
            inRange(hsv_roi, lower_red, upper_red, red_tl_mask1);
            inRange(hsv_roi, lower_red2, upper_red2, red_tl_mask2);
            inRange(hsv_roi, lower_green, upper_green, green_tl_mask);

            Mat red_tl_mask = red_tl_mask1 | red_tl_mask2;

            vector<vector<Point>> red_tl_contours, green_tl_contours;
            findContours(red_tl_mask, red_tl_contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
            findContours(green_tl_mask, green_tl_contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

            for (const auto &contour : red_tl_contours)
            {
                double area = contourArea(contour);
                double perimeter = arcLength(contour, true);
                double circularity = 4 * M_PI * area / (perimeter * perimeter);

                if (circularity > 0.1)
                {
                    rectangle(combinedFrame, tl, Scalar(0, 255, 255), 2);
                    putText(combinedFrame, "Red Light", Point(tl.x, tl.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
                }
            }

            for (const auto &contour : green_tl_contours)
            {
                double area = contourArea(contour);
                double perimeter = arcLength(contour, true);
                double circularity = 4 * M_PI * area / (perimeter * perimeter);

                if (circularity > 0.2)
                {
                    rectangle(combinedFrame, tl, Scalar(0, 255, 0), 2);
                    putText(combinedFrame, "Green Light", Point(tl.x, tl.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
                }
            }
        }

        // Calculate FPS
        frame_count++;
        auto current_time = steady_clock::now();
        double elapsed_seconds = duration_cast<duration<double>>(current_time - start_time).count();
        double fps = frame_count / elapsed_seconds;

        putText(combinedFrame, "FPS: " + to_string(fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        // Write the processed frame to the output video
        outputVideo.write(combinedFrame);

        // Display the processed frame in real-time
        imshow("Processed Frame", combinedFrame);
        if (waitKey(30) == 27)
            break; // ESC key to exit
    }

    for (int i = 0; i < 4; ++i)
    {
        threads[i].join();
    }

    cap.release();
    outputVideo.release();
    destroyAllWindows();

    return 0;
}

