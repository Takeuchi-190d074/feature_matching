#include <opencv2/opencv.hpp>
#include <iostream>

/*  特徴点マッチングによる類似度計算及び入力画像の補正(射影変換)を行う関数 
    src1 : 入力画像(変形される)
    src2 : 入力画像
    dst  : 出力画像(src1をsrc2に合わせて変形した画像)
    
    動作確認済
    1. 特徴点検出:ORB   + 特徴点マッチング:総当たり
    2. 特徴点検出:AKAZE + 特徴点マッチング:総当たり
    
    必要処理時間: 1 < 2 
    精度        : 1 < 2    */

void feature_matching(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst)
{
  std::vector<cv::KeyPoint> key1, key2; // 特徴点を格納
  cv::Mat des1, des2; // 特徴量記述の計算
  const float THRESHOLD = 100; // 類似度の閾値
  float sim = 0;

  /* 比較のために複数手法を記述 必要に応じてコメントアウト*/
  /* 特徴点検出*/
  /* AKAZE */
  cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
  akaze->detect(src1, key1);
  akaze->detect(src2, key2);
  akaze->compute(src1, key1, des1); 
  akaze->compute(src2, key2, des2);
  /* ORB */
  // cv::Ptr<cv::ORB> orb = cv::ORB::create();
  // orb->detect(src1, key1);
  // orb->detect(src2, key2);
  // orb->compute(src1, key1, des1); 
  // orb->compute(src2, key2, des2);

  //std::cout << des1 << std::endl;

  /* 特徴点マッチングアルゴリズム */
  /* 総当たり */
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

  /* 特徴点マッチング */
  std::vector<cv::DMatch> match, match12, match21;
  matcher->match(des1, des2, match);

  /* 特徴量距離の小さい順にソートし、不要な点を削除 */
  for(int i = 0; i < match.size(); i++){
    double min = match[i].distance;
    int n = i;
    for(int j = i + 1; j < match.size(); j++){
      if(min > match[j].distance){
        n = j;
        min = match[j].distance;
      }
    }
    std::swap(match[i], match[n]);
  }
  match.erase(match.begin() + 50, match.end());

  /* 類似度計算(距離による実装、0に近い値ほど画像が類似) */
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

  // cv::drawMatches(src1, key1, src2, key2, match, dst);

  /* src1をsrc2に合わせる形で射影変換して補正 */
  std::vector<cv::Vec2f> get_pt1(match.size()), get_pt2(match.size()); // 使用する特徴点
  /* 対応する特徴点の座標を取得・格納*/
  for(int i = 0; i < match.size(); i++){
    get_pt1[i][0] = key1[match[i].queryIdx].pt.x;
    get_pt1[i][1] = key1[match[i].queryIdx].pt.y;
    get_pt2[i][0] = key2[match[i].trainIdx].pt.x;
    get_pt2[i][1] = key2[match[i].trainIdx].pt.y;
  }

  /* ホモグラフィ行列推定 */
  cv::Mat H = cv::findHomography(get_pt1, get_pt2, cv::RANSAC); 
  /* src1を変形 */
  cv::warpPerspective(src1, dst, H, src2.size());
}