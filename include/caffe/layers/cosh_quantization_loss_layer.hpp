#ifndef CAFFE_COSH_QUANTIZATION_LOSS_LAYER_HPP_
#define CAFFE_COSH_QUANTIZATION_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{
/*
** @ brief to model the hashing quantization loss
**   log(cosh(|z|-1))
**   Added by Fuchen Long in 8/9/2016
*/

template <typename Dtype>
class CoshQuantizationLossLayer : public LossLayer<Dtype>{
public:
	explicit CoshQuantizationLossLayer(const LayerParameter& param)
		: LossLayer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "CoshQuantizationLoss"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	// copydoc CoshQuantization loss
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	int dim_;
	int batch_;
};
}// namespace caffe

#endif