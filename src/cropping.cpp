#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

void cropping(const cv::Mat &src, cv::Mat &dst)
{ 
const static int THRESH_VALUE = 200; // 二値化するときの閾値
cv::Mat gray, ret, ret2;

/* 輪郭抽出の下準備(二値化とオープニング/クロージング) */
cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
cv::adaptiveThreshold(gray, ret, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 5);
cv::dilate(ret, ret, cv::Mat(), cv::Point(-1,-1), 1);
cv::erode(ret, ret, cv::Mat(), cv::Point(-1,-1), 1);

/* 最も外側の輪郭を抽出 */
std::vector<std::vector<cv::Point>> contours; // 抽出時点の頂点
cv::findContours(ret, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
cv::cvtColor(ret, ret, cv::COLOR_GRAY2BGR);
src.convertTo(ret2, CV_32F, 1.0/255);
/* 輪郭を図形に近似 */
std::vector<cv::Point2f> approx; // 輪郭の頂点
for(auto contour = contours.begin(); contour != contours.end(); contour++){
  double epsilon = 0.08 * cv::arcLength(*contour, true);
  cv::approxPolyDP(cv::Mat(*contour), approx, epsilon, true);
  double area = cv::contourArea(approx);
  if(area < src.rows * src.cols / 2){
    dst = src;
    continue;
  }
  //cv::polylines(ret, approx, true, cv::Scalar(0, 0, 255), 2);
  //cv::drawContours(ret, *contour, -1, cv::Scalar(0, 0, 255), cv::LINE_AA);
}
/* 四角形の頂点が検出できていない場合は終了 */
if(approx.size() != 4){
  dst = src;
  return;
}
/* 透視変換 */
std::vector<cv::Point2f> src_vertex = {approx[0], approx[1], approx[2], approx[3]};
int dst_width  = std::hypot(approx[3].x - approx[0].x, approx[3].y - approx[0].y);
int dst_height = std::hypot(approx[1].x - approx[0].x, approx[1].y - approx[0].y);
std::vector<cv::Point2f> dst_vertex = {{0,0}, {0,(float)dst_height}, {(float)dst_width,(float)dst_height}, {(float)dst_width,0}};
dst = cv::Mat::zeros(cv::Size(dst_width, dst_height), CV_32F);
cv::Mat H = cv::getPerspectiveTransform(src_vertex, dst_vertex);
std::cout << "check" << std::endl;
cv::warpPerspective(src, dst, H, dst.size());
}