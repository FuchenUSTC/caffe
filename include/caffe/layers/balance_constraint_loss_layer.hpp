#ifndef CAFFE_BALANCE_CONSTRAINT_LOSS_LAYER_HPP
#define CAFFE_BALANCE_CONSTRAINT_LOSS_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{
/*
** @ briefly to make the hash bits more balance 
**   for balance cosntraint.
**   Added by Fuchen Long in 8/3/2016
*/

template <typename Dtype>
class BalanceConstraintLossLayer :public LossLayer<Dtype>{
public: 
	explicit BalanceConstraintLossLayer(const LayerParameter& param)
		:LossLayer<Dtype>(param), diff_(){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "BalanceConstraintLoss"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	// copydoc BalanceConstraintLossLayer
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	int dim_;
	Dtype lamda_;
	int batch_;
	Blob<Dtype> pow_diff_;
	Blob<Dtype> diff_;
};

}


#endif