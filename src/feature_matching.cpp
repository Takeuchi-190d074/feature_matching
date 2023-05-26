#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>

void feature_matching(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst)
{
  std::vector<cv::KeyPoint> key1, key2; // 特徴点を格納
  cv::Mat des1, des2; // 特徴量記述の計算
  const float THRESHOLD = 100; // 類似度の閾値
  float sim = 0;

  /* 比較のために複数手法を記述 必要に応じてコメントアウト*/
  /* 特徴点検出*/
  /* AKAZE */
  // cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f);
  // akaze->detect(src1, key1);
  // akaze->detect(src2, key2);
  // akaze->compute(src1, key1, des1); 
	// akaze->compute(src2, key2, des2);
  /* ORB */
	cv::Ptr<cv::ORB> orb = cv::ORB::create(100);
	orb->detect(src1, key1);
	orb->detect(src2, key2);
  orb->compute(src1, key1, des1); 
	orb->compute(src2, key2, des2);
  // std::cout << des1 << std::endl;

  /* 特徴点マッチングアルゴリズム */
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

  /* 特徴点マッチング */
  /* クロスチェックを行い、両方でマッチしたものだけ残す */
  std::vector<cv::DMatch> match, match12, match21;
	matcher->match(des1, des2, match12);
  matcher->match(des2, des1, match21);
  for(int i = 0; i < match12.size(); i++){
    cv::DMatch forward = match12[i];
    cv::DMatch backward = match21[forward.trainIdx];
    if (backward.trainIdx == forward.queryIdx){
      match.push_back(forward);
    }
  }
  cv::drawMatches(src1, key1, src2, key2, match, dst);

  for(int i = 0; i < match.size(); i++){
    cv::DMatch dis = match[i];
    sim += dis.distance;
  }
  sim /= match.size();
  std::cout << "類似度: " << sim << std::endl; 


  /* 画像の類似度が低すぎる場合は終了 */
  if(0/* sim > THRESHOLD*/){
    std::cerr << "画像が違いすぎます" << std::endl;
    std::exit(1);
  }

}