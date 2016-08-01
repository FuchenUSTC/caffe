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
		<< "Video Temporal Pooling layer: Frame Number should be divided.";
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
	int frame_num){
	int res_offset = 0;
	int des_offset = 0;
	for (int n = 0; n < des_shape[0]; ++n){
		for (int c = 0; c < des_shape[1]; ++c){
			for (int w = 0; w < des_shape[3]; ++w){
				res_offset = ((n*frame_num) + w)*res_shape[1] + c;

				des[des_offset] = res[res_offset];
				des_offset += 1;
			}
		}
	}
}

template <typename Dtype>
void video_pooling_backward_cpu(const Dtype* res, Dtype* des,
	const vector<int> res_shape, const vector<int> des_shape,
	int frame_num){
	int res_offset = 0;
	int des_offset = 0;
	for (int n = 0; n < res_shape[0]; ++n){
		for (int c = 0; c < res_shape[1]; ++c){
			for (int w = 0; w < res_shape[3]; ++w){
				des_offset = ((n*frame_num) + w)*des_shape[1] + c;

				des[des_offset] = res[res_offset];
				res_offset += 1;
			}
		}
	}
}

template <typename Dtype>
void VideoTemporalPoolingLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	video_pooling_forward_cpu(bottom_data, top_data,
		bottom[0]->shape(), top[0]->shape(), frame_num_);
}

template <typename Dtype>
void VideoTemporalPoolingLayer<Dtype>::Backward_cpu(
	const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom ){
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	if (propagate_down[0]){
		video_pooling_backward_cpu(top_diff, bottom_diff,
			top[0]->shape(), bottom[0]->shape(), frame_num_);
	}
}
 
INSTANTIATE_CLASS(VideoTemporalPoolingLayer);
REGISTER_LAYER_CLASS(VideoTemporalPooling);
} // namespace caffe