#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/video_temporal_pooling_layer.hpp"

// added by Fuchen Long for video hash coding
// learning 

namespace caffe{
template <typename Dtype>
void VideoTemporalPoolingLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	frame_num_ = this->layer_param_.video_temporal_pooling_param().frame_num();
	CHECK_EQ(bottom[0]->shape(0) % frame_num_, 0)
		<< "Video Temporal Pooling layer: Frame Number should be divied.";
}

template <typename Dtype>
void VideoTemporalPoolingLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	vector<int> bottom_shape = bottom[0]->shape();
	vector<int> top_shape;

	top_shape = vector<int>(4, 0);
	top_shape[0] = bottom_shape[0] / frame_num_;
	top_shape[1] = bottom_shape[1];
	top_shape[2] = 1;
	top_shape[3] = frame_num_;

	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void video_pooling_forward_cpu(const Dtype*res, Dtype* des,
	const vector<int> res_shape, const vector<int> des_shape,
	int frame_num, int inner){
	int res_offset = 0;
	int des_offset = 0;
	for (int n = 0; n < res_shape[0]; ++n){
		for (int c = 0; c < res_shape[1]; ++c){
			for (int w = 0; w < res_shape[3]; ++w){
				des_offset = (n*frame_num + w)*res_shape[3] + c;

				caffe_copy(1, res + res_offset, des + des_offset);
				res_offset += 1;
			}
		}
	}
}


} // namespace caffe