#include <algorithm>
#include <vector>

#include "caffe/layers/balance_constraint_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// added by Fuchen Long in 8/3/2016
namespace caffe{

template <typename Dtype>
void BalanceConstraintLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	dim_ = this->layer_param_.balance_constraint_loss_param().dim();
	lamda_ = this->layer_param_.balance_constraint_loss_param().lamda();
	batch_ = bottom[0]->num();
	CHECK_EQ(bottom[0]->channels(), dim_)
		<< "BalanceConstraintLossLayer: code length must match.";
	pow_diff_.Reshape(batch_, dim_, 1, 1);
	diff_.Reshape(batch_, 1, 1, 1);
}

template <typename Dtype>
void BalanceConstraintLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	CHECK_EQ(bottom[0]->channels(), dim_)
		<< "BalanceConstraintLossLayer: code length must match.";
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void BalanceConstraintLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* diff_data = diff_.mutable_cpu_data();
	Dtype balanceloss(0.0);
	int dim_all_ = batch_*dim_;

	caffe_powx(dim_all_, bottom_data, Dtype(2.0), pow_diff_.mutable_cpu_data());
	const Dtype* pow_data = pow_diff_.cpu_data();
	for (int n = 0; n < batch_; ++n){
		int offset = n*dim_;
		Dtype loss(0.0);
		loss = caffe_cpu_asum(dim_, pow_data + offset) - lamda_*dim_;
		balanceloss += loss*loss;
		diff_data[n] = loss;
	}
	top[0]->mutable_cpu_data()[0] = balanceloss / batch_;
}

template <typename Dtype>
void BalanceConstraintLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* diff_data = diff_.cpu_data();
	int dim_all_ = batch_*dim_;

	if (propagate_down[0]){
		caffe_copy(dim_all_, bottom[0]->cpu_data(), bottom_diff);
		for (int n = 0; n < batch_; ++n){
			int offset = n*dim_;
			caffe_scal(dim_, Dtype(8.0)*diff_data[n] / batch_, bottom_diff + offset);
		}
	}
}

INSTANTIATE_CLASS(BalanceConstraintLossLayer);
REGISTER_LAYER_CLASS(BalanceConstraintLoss);
}