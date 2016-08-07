#include <cmath>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/weight_plus_layer.hpp"


namespace caffe{

/**
 * Weight plus layer for hashing learning
 * Added by Fuchen Long in 8/7/2016
 * in this opeartion, some matrix transfer will added 
*/

template <typename Dtype>
void WeightPlusLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	batch_ = bottom[0]->num();
	dim_ = this->layer_param_.weight_plus_param().dim();
	CHECK_EQ(bottom[0]->channels(), dim_)
		<< "Weight Plus Layer: the codelenght should match.";
	this->blobs_.resize(1); // for the scale hashing
	vector<int> weight_shape(1);
	weight_shape[0] = dim_; 
	this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
	shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
		this->layer_param_.weight_plus_param().weight_filler()));
	weight_filler->Fill(this->blobs_[0].get()); // the weight is 1 first
	this->param_propagate_down_.resize(this->blobs_.size(), true);
	weight_pow_.Reshape(dim_, 1, 1, 1);
	weight_two_.Reshape(dim_, 1, 1, 1);
	data_meta_.Reshape(batch_, dim_, 1, 1);
}

template <typename Dtype>
void WeightPlusLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	for (int n = 0; n < batch_; ++n){
		int offset = n*dim_;
		caffe_powx(dim_, weight, Dtype(2.0), weight_pow_.mutable_cpu_data());
		caffe_mul(dim_, bottom_data + offset, weight_pow_.cpu_data(), top_data + offset);
	}
}


template <typename Dtype>
void WeightPlusLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	caffe_scal(dim_, Dtype(2.0), weight_two_.mutable_cpu_data());

	// gradient with respect to weight
	for (int n = 0; n < batch_; ++n){
		int offset = n*dim_;
		caffe_mul(dim_, weight_two_.cpu_data(), bottom_data + offset, data_meta_.mutable_cpu_data() + offset);
		caffe_mul(dim_, top_diff + offset, data_meta_.cpu_data() + offset, data_meta_.mutable_cpu_data() + offset);
		caffe_axpy(dim_, Dtype(1.0), data_meta_.cpu_data() + offset, blobs_[0]->mutable_cpu_diff());
	}

	// gradient with respect to bottom data
	if (propagate_down[0]){
		for (int n = 0; n < batch_; ++n){
			int offset = n*dim_;
			caffe_mul(dim_, top_diff + offset, weight_two_.cpu_data(), bottom_diff + offset);
		}
	}

}

INSTANTIATE_CLASS(WeightPlusLayer);
REGISTER_LAYER_CLASS(WeightPlus);

}// namespace caffe