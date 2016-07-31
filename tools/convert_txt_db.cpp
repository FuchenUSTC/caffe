// This program converts a set of feature to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//    convert_txt_db.exe [FLAGS] FEATURE_LIST OUTPUT_DB FEATURE_NUM DIM
//  
//
//   ....


// added by Fuchen Long in 7/31/2016

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "boost/scoped_ptr.hpp"
#include "stdint.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"



using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
DEFINE_string(backend, "leveldb",
	"The backend {lmdb, leveldb} for storing the result");

int main(int args, char** argv){ // usage: convert_binaryfea_lvdb feature_bin outputdbnum labellist
	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Convert a set of txt features to leveldb/lmdb\n"
		"for the hashing/feature learning.\n"
		"Usage:\n"
		"    convert_txt_db.exe [FLAGS] FEATURE_LIST OUTPUT_DB FEATURE_NUM DIM <LABEL_PATH>\n"
		"\n"
		);
	gflags::ParseCommandLineFlags(&args, &argv, true);

	if (args < 5){
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_triplet_feature_data");
		return 1;
	}

	// set the buff size of each lvdb
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[2], db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());


	const int kMaxKeyLength = 100;
	char key[kMaxKeyLength];
	std::string value;

	// read the file of the binary files 
	std::ifstream feature_in(argv[1], ios::in); // to set the number of the feature
	std::vector<int> label;

	int feature_num = atoi(argv[3]);
	int feature_dim = atoi(argv[4]);


	// set the feature label
	if (args == 6){
		std::ifstream label_in(argv[5], ios::in); // to set the label
		for (int i = 0; i < feature_num; ++i){
			int index;
			label_in >> index;
			label.push_back(index);
		}
		label_in.close();
	}
	else{
		for (int i = 0; i < feature_num; ++i) label.push_back(0);
	}
	int count = 0;

	// And to read the next feature for the training
	for (int i = 0; i < feature_num; ++i){
		caffe::Datum datum;
		datum.set_channels(feature_dim);
		datum.set_height(1);
		datum.set_width(1);
		float feature;
		for (int j = 0; j < feature_dim; ++j){
			feature_in >> feature;
			datum.add_float_data(feature);
		}
		datum.set_label(label[i]);
		datum.SerializePartialToString(&value);
		int length = _snprintf(key, kMaxKeyLength, "%08d", i);
		txn->Put(std::string(key,length), value);
		if (++count % 1000 == 0) LOG(ERROR) << "Have converted " << count << " txt feature.\n";
	}

	if (count % 1000 != 0){
		txn->Commit();
		LOG(ERROR) << "Success processing " << count << " files.";
	}

	// delete the db
	feature_in.close();
}