#ifndef BMAX_H
#define BMAX_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>

class BMAX
{
public:
    BMAX();
    void createVocabulary(cv::Mat samples, int words);
    void setVocabulary(cv::Mat vocabulary);
    void train(cv::Mat samples, cv::Mat labels);
    void train(cv::Mat samples, cv::Mat labels, int words);
    void train(std::vector<cv::Mat> images, cv::Mat labels, int words);

    int classifyImage(cv::Mat image);
    int classify(cv::Mat points);

private:
    cv::Mat vocabulary;
    cv::SVM svm;

    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor;
    cv::Ptr<cv::DescriptorMatcher> matcher;

};

#endif // BMAX_H
