#include <vector>
#include <cstring>
#include <time.h>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/layers/residual_quantization_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

// added by Fuchen Long for 
// testing the residual quantization layer

namespace caffe{

template <typename TypeParam>
class ResidualQuantizationLayerTest : public MultiDeviceTest<TypeParam>{
	typedef typename TypeParam::Dtype Dtype;
protected:
	ResidualQuantizationLayerTest()
		: blob_bottom_code_(new Blob<Dtype>(12, 128, 1, 1)),
		blob_bottom_loss_(new Blob<Dtype>(12, 128, 1, 1)),
		blob_top_(new Blob<Dtype>()){
		// init the input code and loss
		Dtype* bottom_code = blob_bottom_code_->mutable_cpu_data();
		Dtype* bottom_loss = blob_bottom_loss_->mutable_cpu_data();
		std::srand(time(NULL));
		for (int i = 0; i < 12*128; ++i) {
			bottom_code[i] = (std::rand() % 100000)*0.00001;
			bottom_loss[i] = (std::rand() % 100000)*0.00001;
		}
		blob_bottom_vec_.push_back(blob_bottom_code_);
		blob_bottom_vec_.push_back(blob_bottom_loss_);
		blob_top_vec_.push_back(blob_top_);
	}

	virtual ~ResidualQuantizationLayerTest(){
		delete blob_bottom_code_;
		delete blob_bottom_loss_;
		delete blob_top_;
	}

	void TestForward(){
		// Set the paramter and check the value
		LayerParameter layer_param;
		ResidualQuantizationParameter* res_param =
			layer_param.mutable_residual_quantization_param();
		res_param->set_lamda(1.0);
		ResidualQuantizationLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	}



	Blob<Dtype>* blob_bottom_code_;
	Blob<Dtype>* blob_bottom_loss_;
	Blob<Dtype>* blob_top_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ResidualQuantizationLayerTest, TestDtypesAndDevices);

TYPED_TEST(ResidualQuantizationLayerTest, TestForward){
	this->TestForward();
}

TYPED_TEST(ResidualQuantizationLayerTest, TestGradient){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	ResidualQuantizationParameter* res_param =
		layer_param.mutable_residual_quantization_param();
	res_param->set_lamda(1.0);
	ResidualQuantizationLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	GradientChecker<Dtype> checker(1e-5, 1e-2);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_);
}
}