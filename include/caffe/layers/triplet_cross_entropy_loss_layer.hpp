#ifndef CAFFE_TRIPLET_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{

/*
*  @ added by Fuchen Long, used in hashing learning 
*    and this is inspired by the Deep hashing work 
*    in AAAI16.
*    Added in 8/19/2016
*/

template<typename Dtype>
class TripletCrossEntropyLossLayer :public LossLayer<Dtype>{
public:
	explicit TripletCrossEntropyLossLayer(const LayerParameter& param)
		:LossLayer<Dtype>(param), diff_(){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline int ExactNumBottomBlobs() const { return 3; }
	virtual inline const char* type() const { return "TripletCrossEntropyLoss"; }
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return bottom_index != 3;
	}

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int dim_;
	int batch_;
	Blob<Dtype> diff_;
	Blob<Dtype> inner_product_q_s_;
	Blob<Dtype> inner_product_q_d_;
	Blob<Dtype> exp_qmeta_q_s_;
	Blob<Dtype> exp_qmeta_q_d_;
	Blob<Dtype> gradient_q_;
	Blob<Dtype> exp_smeta_q_s_;
	Blob<Dtype> gradient_s_;
	Blob<Dtype> gradient_d_;
};

}// namespace caffe

#endif