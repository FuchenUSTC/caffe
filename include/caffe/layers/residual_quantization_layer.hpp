#ifndef CAFFE_RESIDUAL_QUANTIZATION_LAYER_HPP_
#define CAFFE_RESIDUAL_QUANTIZATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{

/**
 * @brief Residual Quantization loss
 *  for the hashing learning, the residual method to
 *  model the hashing quantization loss
 *  added by Fuchen Long in 8/7/2016
 */

template <typename Dtype>
class ResidualQuantizationLayer : public NeuronLayer<Dtype>{
public:
	explicit ResidualQuantizationLayer(const LayerParameter& param)
		: NeuronLayer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "ResidualQuantization"; }
	virtual inline int ExactNumBottomBlobs() const { return 2; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	void quantization_add_forward_cpu_sig(const Dtype* code, const Dtype* loss,
		Dtype* res_code);
	void quantization_add_forward_cpu_tanh(const Dtype* code, const Dtype* loss,
		Dtype* res_code);
	void quantization_add_backward_cpu_sig(const Dtype* code, const Dtype* loss,
		const Dtype* top_diff, const Dtype* res_code,
		Dtype* code_diff, Dtype* loss_diff);
	void quantization_add_backward_cpu_tanh(const Dtype* code, const Dtype* loss,
		const Dtype* top_diff, const Dtype* res_code,
		Dtype* code_diff, Dtype* loss_diff);

	int batch_;
	int dim_;
	bool sigmoid_flag_;
};
}// namespace caffe

#endif