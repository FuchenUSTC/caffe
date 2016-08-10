#ifndef CAFFE_LNORM_LOSS_LAYER_HPP
#define CAFFE_LNORM_LOSS_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{

/*
** @ brief to model the L2 Loss
**   for the quantization loss modeling
**   Added by Fuchen Long in 8/10/2016
*/

template <typename Dtype>
class LnormLossLayer : public LossLayer<Dtype>{
public:
	explicit LnormLossLayer(const LayerParameter& param)
		: LossLayer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "LnormLoss"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	// copydoc LnormLossLayer loss
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int dim_;
	int batch_;
	Blob<Dtype> diff_meta_;
};
}// namespace caffe


#endif