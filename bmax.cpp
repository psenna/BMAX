#include "bmax.h"

BMAX::BMAX()
{

}

int BMAX::classify(cv::Mat points){

}

int BMAX::classifyImage(cv::Mat image){

}

void BMAX::train(cv::Mat samples, cv::Mat labels){

}

void BMAX::train(cv::Mat samples, cv::Mat labels, int words){

}

void BMAX::train(std::vector<cv::Mat> images, cv::Mat labels, int words){

}

void BMAX::createVocabulary(cv::Mat samples, int words){

}

void BMAX::setVocabulary(cv::Mat vocabulary){
    this->vocabulary = vocabulary.clone();
}
