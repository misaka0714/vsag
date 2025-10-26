
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pq_fastscan_quantizer_parameter.h"

#include "impl/logger/logger.h"
#include "inner_string_params.h"

namespace vsag {

PQFastScanQuantizerParameter::PQFastScanQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_PQFS) {
}

void
PQFastScanQuantizerParameter::FromJson(const JsonType& json) {
    if (json.Contains(PRODUCT_QUANTIZATION_DIM) &&
        json[PRODUCT_QUANTIZATION_DIM].IsNumberInteger()) {
        this->pq_dim_ = json[PRODUCT_QUANTIZATION_DIM].GetInt();
    }
    
    if (json.Contains(PRODUCT_QUANTIZATION_TRAIN_SAMPLE_SIZE) &&
        json[PRODUCT_QUANTIZATION_TRAIN_SAMPLE_SIZE].IsNumberInteger()) {
        this->train_sample_size_ = json[PRODUCT_QUANTIZATION_TRAIN_SAMPLE_SIZE].GetInt();
    }
}

JsonType
PQFastScanQuantizerParameter::ToJson() const {
    JsonType json;
    json[QUANTIZATION_TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_PQFS);
    json[PRODUCT_QUANTIZATION_DIM].SetInt(this->pq_dim_);
    json[PRODUCT_QUANTIZATION_TRAIN_SAMPLE_SIZE].SetInt(this->train_sample_size_);
    return json;
}

bool
PQFastScanQuantizerParameter::CheckCompatibility(const ParamPtr& other) const {
    auto pq_fast_param = std::dynamic_pointer_cast<const PQFastScanQuantizerParameter>(other);
    if (not pq_fast_param) {
        logger::error(
            "PQFastScanQuantizerParameter::CheckCompatibility: "
            "other is not PQFastScanQuantizerParameter");
        return false;
    }
    if (this->pq_dim_ != pq_fast_param->pq_dim_) {
        logger::error(
            "PQFastScanQuantizerParameter::CheckCompatibility: "
            "pq_dim mismatch, this: {}, other: {}",
            this->pq_dim_,
            pq_fast_param->pq_dim_);
        return false;
    }
    if (this->train_sample_size_ != pq_fast_param->train_sample_size_) {
        logger::error(
            "PQFastScanQuantizerParameter::CheckCompatibility: "
            "train_sample_size mismatch, this: {}, other: {}",
            this->train_sample_size_,
            pq_fast_param->train_sample_size_);
        return false;
    }
    return true;
}
}  // namespace vsag
