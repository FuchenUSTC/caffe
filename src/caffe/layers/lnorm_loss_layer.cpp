#include <algorithm>
#include <vector>

#include "caffe/layers/lnorm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// added by Fuchen Long in 8/10/2016
namespace caffe{

template <typename Dtype>
void LnormLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	dim_ = bottom[0]->channels();
	batch_ = bottom[0]->num();
	CHECK_EQ(bottom[0]->channels(), dim_)
		<< ": code length must match.";
	diff_meta_.Reshape(dim_, batch_, 1, 1);
}

template <typename Dtype>
void LnormLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	CHECK_EQ(bottom[0]->channels(), dim_)
		<< "LnormLossLayer: code length must match.";
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void LnormLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	caffe_powx(dim_*batch_, bottom_data, Dtype(2.0), diff_meta_.mutable_cpu_data());
	Dtype lnorm_loss = caffe_cpu_asum(dim_*batch_, diff_meta_.cpu_data());
	top[0]->mutable_cpu_data()[0] = lnorm_loss / batch_;
}

template <typename Dtype>
void LnormLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	if (propagate_down[0]){
		caffe_copy(dim_*batch_, bottom_data, bottom_diff);
		caffe_scal(dim_*batch_, Dtype(4.0) / batch_, bottom_diff);
	}
}

INSTANTIATE_CLASS(LnormLossLayer);
REGISTER_LAYER_CLASS(LnormLoss);
}// namespace caffe