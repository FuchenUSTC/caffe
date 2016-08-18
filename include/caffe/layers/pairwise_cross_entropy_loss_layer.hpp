#ifndef CAFFE_PAIRWISE_CROSS_ENTROPY_LOSS_HPP_
#define CAFFE_PARIWISE_CROSS_ENTROPY_LOSS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{

/*
*  @ added by Fuchen Long, used in hashing learning
*    and to implement the baseline for the paper
*    Deep Hashing Network for Efficient Similarity Retrieval (AAAI16)
*    Added in 8/19/2016
*/

template<typename Dtype>
class PairWiseCrossEntropyLossLayer :public LossLayer<Dtype>{
public:
	explicit PairWiseCrossEntropyLossLayer(const LayerParameter& param)
		:LossLayer<Dtype>(param), diff_(){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline int ExactNumBottomBlobs() const { return 3; }
	virtual inline const char* type() const { return "PairWiseCrossEntropyLoss"; }
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
	Dtype lamda_;
	Blob<Dtype> diff_;
	Blob<Dtype> inner_product_;
	Blob<Dtype> meta_data_;
	Blob<Dtype> inner_exp_;
	Blob<Dtype> inner_log_;
	Blob<Dtype> gradient_1_;
	Blob<Dtype> gradient_2_;
	Blob<Dtype> meta_g_1_;
	Blob<Dtype> meta_g_2_;
};

}// namespace caffe


#endif